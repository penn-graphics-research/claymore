#ifndef __MGSP_BENCHMARK_CUH_
#define __MGSP_BENCHMARK_CUH_
#include "grid_buffer.cuh"
#include "hash_table.cuh"
#include "mgmpm_kernels.cuh"
#include "particle_buffer.cuh"
#include "settings.h"
#include <MnBase/Concurrency/Concurrency.h>
#include <MnBase/Meta/ControlFlow.h>
#include <MnBase/Meta/TupleMeta.h>
#include <MnBase/Profile/CppTimers.hpp>
#include <MnBase/Profile/CudaTimers.cuh>
#include <MnSystem/Cuda/Cuda.h>
#include <MnSystem/IO/IO.h>
#include <MnSystem/IO/ParticleIO.hpp>
#include <array>
#include <fmt/color.h>
#include <fmt/core.h>
#include <vector>

namespace mn {

using iojob = tuple<std::string, std::vector<std::array<float, 3>>>;

struct mgsp_benchmark {
  using streamIdx = Cuda::StreamIndex;
  using eventIdx = Cuda::EventIndex;
  struct device_allocator { // hide the global one
    void *allocate(std::size_t bytes) {
      void *ret;
      checkCudaErrors(cudaMalloc(&ret, bytes));
      return ret;
    }
    void deallocate(void *p, std::size_t) { checkCudaErrors(cudaFree(p)); }
  };
  template <std::size_t I> void initParticles() {
    auto &cuDev = Cuda::ref_cuda_context(I);
    cuDev.setContext();
    tmps[I].alloc(config::g_max_active_block);
    for (int copyid = 0; copyid < 2; copyid++) {
      gridBlocks[copyid].emplace_back(device_allocator{});
      particleBins[copyid].emplace_back(
          ParticleBuffer<get_material_type(I)>{device_allocator{}});
      partitions[copyid].emplace_back(device_allocator{},
                                      config::g_max_active_block);
    }
    particles[I] = spawn<particle_array_, orphan_signature>(device_allocator{});
    checkedCnts[I][0] = 0;
    checkedCnts[I][1] = 0;
    curNumActiveBlocks[I] = config::g_max_active_block;
    curNumActiveBins[I] = config::g_max_particle_bin;
    /// tail-recursion optimization
    if constexpr (I + 1 < config::g_device_cnt)
      initParticles<I + 1>();
  }
  mgsp_benchmark()
      : dtDefault{1e-4}, curTime{0.f}, rollid{0}, curFrame{0}, curStep{0},
        fps{24}, bRunning{true} {
    // data
    initParticles<0>();
    fmt::print("{} -vs- {}\n",
               match(particleBins[0][0])([&](auto &pb) { return pb.size; }),
               match(particleBins[0][1])([&](auto &pb) { return pb.size; }));
    // tasks
    for (int did = 0; did < config::g_device_cnt; ++did) {
      ths[did] = std::thread([this](int did) { this->gpu_worker(did); }, did);
    }
  }
  ~mgsp_benchmark() {
    auto is_empty = [this]() {
      for (int did = 0; did < config::g_device_cnt; ++did)
        if (!jobs[did].empty())
          return false;
      return true;
    };
    do {
      cv_slave.notify_all();
    } while (!is_empty());
    bRunning = false;
    for (auto &th : ths)
      th.join();
  }

  void initModel(int devid, const std::vector<std::array<float, 3>> &model) {
    auto &cuDev = Cuda::ref_cuda_context(devid);
    cuDev.setContext();
    pcnt[devid] = model.size();
    fmt::print("init model[{}] with {} particles\n", devid, pcnt[devid]);
    cudaMemcpyAsync((void *)&particles[devid].val_1d(_0, 0), model.data(),
                    sizeof(std::array<float, 3>) * model.size(),
                    cudaMemcpyDefault, cuDev.stream_compute());
    cuDev.syncStream<streamIdx::Compute>();

    std::string fn = std::string{"model"} + "_dev[" + std::to_string(devid) +
                     "]_frame[0].bgeo";
    IO::insert_job([fn, model]() { write_partio<float, 3>(fn, model); });
    IO::flush();
  }
  template <typename CudaContext>
  void exclScan(std::size_t cnt, int const *const in, int *out,
                CudaContext &cuDev) {
#if 1
    thrust::exclusive_scan(getDevicePtr(in), getDevicePtr(in) + cnt,
                           getDevicePtr(out));
#else
    std::size_t temp_storage_bytes = 0;
    auto plus_op = [] __device__(const int &a, const int &b) { return a + b; };
    checkCudaErrors(cub::DeviceScan::ExclusiveScan(nullptr, temp_storage_bytes,
                                                   in, out, plus_op, 0, cnt,
                                                   cuDev.stream_compute()));
    void *d_tmp = tmps[cuDev.getDevId()].d_tmp;
    checkCudaErrors(cub::DeviceScan::ExclusiveScan(d_tmp, temp_storage_bytes,
                                                   in, out, plus_op, 0, cnt,
                                                   cuDev.stream_compute()));
#endif
  }
  float getMass(int did) {
    return match(particleBins[rollid][did])(
        [&](const auto &particleBuffer) { return particleBuffer.mass; });
  }
  void checkCapacity(int did) {
    if (ebcnt[did] > curNumActiveBlocks[did] * 3 / 4 &&
        checkedCnts[did][0] == 0) {
      curNumActiveBlocks[did] = curNumActiveBlocks[did] * 3 / 2;
      checkedCnts[did][0] = 2;
      fmt::print(fmt::emphasis::bold, "resizing blocks {} -> {}\n", ebcnt[did],
                 curNumActiveBlocks[did]);
    }
    if (bincnt[did] > curNumActiveBins[did] * 3 / 4 &&
        checkedCnts[did][1] == 0) {
      curNumActiveBins[did] = curNumActiveBins[did] * 3 / 2;
      checkedCnts[did][1] = 2;
      fmt::print(fmt::emphasis::bold, "resizing bins {} -> {}\n", bincnt[did],
                 curNumActiveBins[did]);
    }
  }
  /// thread local ctrl flow
  void gpu_worker(int did) {
    auto wait = [did, this]() {
      std::unique_lock<std::mutex> lk{this->mut_slave};
      this->cv_slave.wait(lk, [did, this]() {
        return !this->bRunning || !this->jobs[did].empty();
      });
    };
    auto signal = [this]() {
      std::unique_lock<std::mutex> lk{this->mut_ctrl};
      this->idleCnt.fetch_add(1);
      lk.unlock();
      this->cv_ctrl.notify_one();
    };
    auto &cuDev = Cuda::ref_cuda_context(did);
    cuDev.setContext();
    fmt::print(fg(fmt::color::light_blue),
               "{}-th gpu worker operates on GPU {}\n", did, cuDev.getDevId());
    while (this->bRunning) {
      wait();
      auto job = this->jobs[did].try_pop();
      if (job)
        (*job)(did);
      signal();
    }
    fmt::print(fg(fmt::color::light_blue), "{}-th gpu worker exits\n", did);
  }
  void main_loop() {
    auto sync = [this]() {
      std::unique_lock<std::mutex> lk{this->mut_ctrl};
      this->cv_ctrl.wait(
          lk, [this]() { return this->idleCnt == config::g_device_cnt; });
      fmt::print(fmt::emphasis::bold,
                 "-----------------------------------------------------------"
                 "-----\n");
    };
    auto issue = [this](std::function<void(int)> job, uint32_t opt = 0) {
      std::unique_lock<std::mutex> lk{mut_slave};
      for (int did = 0; did < config::g_device_cnt; ++did)
        jobs[did].push(job);
      idleCnt = 0;
      lk.unlock();
      cv_slave.notify_all();
    };
    /// initial
    float nextTime = 1.f / fps;
    dt = compute_dt(0.f, curTime, nextTime, dtDefault);
    fmt::print(fmt::emphasis::bold, "{} --{}--> {}, defaultDt: {})\n", curTime,
               dt, nextTime, dtDefault);
    issue([this](int did) { initial_setup(did); });
    sync();
    curTime = dt;
    for (curFrame = 1; curFrame <= config::g_total_frame_cnt; ++curFrame) {
      for (; curTime < nextTime; curTime += dt, curStep++) {
        /// max grid vel
        issue([this](int did) {
          auto &cuDev = Cuda::ref_cuda_context(did);
          /// check capacity
          checkCapacity(did);
          float *d_maxVel = tmps[did].d_maxVel;
          CudaTimer timer{cuDev.stream_compute()};
          timer.tick();
          checkCudaErrors(cudaMemsetAsync(d_maxVel, 0, sizeof(float),
                                          cuDev.stream_compute()));
          cuDev.compute_launch(
              {(nbcnt[did] + g_num_grid_blocks_per_cuda_block - 1) /
                   g_num_grid_blocks_per_cuda_block,
               g_num_warps_per_cuda_block * 32, g_num_warps_per_cuda_block},
              update_grid_velocity_query_max, (uint32_t)nbcnt[did],
              gridBlocks[0][did], partitions[rollid][did], dt, d_maxVel);
          checkCudaErrors(cudaMemcpyAsync(&maxVels[did], d_maxVel,
                                          sizeof(float), cudaMemcpyDefault,
                                          cuDev.stream_compute()));
          timer.tock(fmt::format("GPU[{}] frame {} step {} grid_update_query",
                                 did, curFrame, curStep));
        });
        sync();
        /// host: compute maxvel & next dt
        float maxVel = 0.f;
        for (int did = 0; did < g_device_cnt; ++did)
          if (maxVels[did] > maxVel)
            maxVel = maxVels[did];
        nextDt = compute_dt(maxVel, curTime, nextTime, dtDefault);
        fmt::print(fmt::emphasis::bold,
                   "{} --{}--> {}, defaultDt: {}, maxVel: {})\n", curTime,
                   nextDt, nextTime, dtDefault, maxVel);
        /// g2p2g
        issue([this](int did) {
          auto &cuDev = Cuda::ref_cuda_context(did);
          CudaTimer timer{cuDev.stream_compute()};

          /// check capacity
          if (checkedCnts[did][1] > 0) {
            match(particleBins[rollid ^ 1][did])([&](auto &pb) {
              pb.resize(device_allocator{}, curNumActiveBins[did]);
            });
            checkedCnts[did][1]--;
          }

          timer.tick();
          // grid
          gridBlocks[1][did].reset(nbcnt[did], cuDev);
          // adv map
          checkCudaErrors(
              cudaMemsetAsync(partitions[rollid][did]._ppcs, 0,
                              sizeof(int) * ebcnt[did] * g_blockvolume,
                              cuDev.stream_compute()));
          // g2p2g
          match(particleBins[rollid][did])([&](const auto &pb) {
            cuDev.compute_launch(
                {pbcnt[did], 128, (512 * 3 * 4) + (512 * 4 * 4)}, g2p2g, dt,
                nextDt, pb,
                get<typename std::decay_t<decltype(pb)>>(
                    particleBins[rollid ^ 1][did]),
                partitions[rollid ^ 1][did], partitions[rollid][did],
                gridBlocks[0][did], gridBlocks[1][did]);
          });
          timer.tock(fmt::format("GPU[{}] frame {} step {} g2p2g", did,
                                 curFrame, curStep));
          if (checkedCnts[did][0] > 0) {
            partitions[rollid ^ 1][did].resizePartition(
                device_allocator{}, curNumActiveBlocks[did]);
            checkedCnts[did][0]--;
          }
        });
        sync();
        issue([this](int did) {
          auto &cuDev = Cuda::ref_cuda_context(did);
          CudaTimer timer{cuDev.stream_compute()};
          timer.tick();
          /// mark particle blocks
          partitions[rollid][did].buildParticleBuckets(cuDev, ebcnt[did]);

          int *activeBlockMarks = tmps[did].activeBlockMarks,
              *destinations = tmps[did].destinations,
              *sources = tmps[did].sources;
          checkCudaErrors(cudaMemsetAsync(activeBlockMarks, 0,
                                          sizeof(int) * nbcnt[did],
                                          cuDev.stream_compute()));
          /// mark grid blocks
          cuDev.compute_launch({(nbcnt[did] * g_blockvolume + 127) / 128, 128},
                               mark_active_grid_blocks, (uint32_t)nbcnt[did],
                               gridBlocks[1][did], activeBlockMarks);
          cuDev.compute_launch({(ebcnt[did] + 1 + 127) / 128, 128},
                               mark_active_particle_blocks, ebcnt[did] + 1,
                               partitions[rollid][did]._ppbs, sources);
          exclScan(ebcnt[did] + 1, sources, destinations, cuDev);
          /// building new partition
          // block count
          checkCudaErrors(cudaMemcpyAsync(
              partitions[rollid ^ 1][did]._cnt, destinations + ebcnt[did],
              sizeof(int), cudaMemcpyDefault, cuDev.stream_compute()));
          checkCudaErrors(cudaMemcpyAsync(
              &pbcnt[did], destinations + ebcnt[did], sizeof(int),
              cudaMemcpyDefault, cuDev.stream_compute()));
          cuDev.compute_launch({(ebcnt[did] + 255) / 256, 256},
                               exclusive_scan_inverse, ebcnt[did],
                               (const int *)destinations, sources);
          // indextable, activeKeys, ppb, buckets
          partitions[rollid ^ 1][did].resetTable(cuDev.stream_compute());
          cuDev.syncStream<streamIdx::Compute>();
          cuDev.compute_launch({pbcnt[did], 128}, update_partition,
                               (uint32_t)pbcnt[did], (const int *)sources,
                               partitions[rollid][did],
                               partitions[rollid ^ 1][did]);
          // binsts
          {
            int *binpbs = tmps[did].binpbs;
            cuDev.compute_launch({(pbcnt[did] + 1 + 127) / 128, 128},
                                 compute_bin_capacity, pbcnt[did] + 1,
                                 (const int *)partitions[rollid ^ 1][did]._ppbs,
                                 binpbs);
            exclScan(pbcnt[did] + 1, binpbs,
                     partitions[rollid ^ 1][did]._binsts, cuDev);
            checkCudaErrors(cudaMemcpyAsync(
                &bincnt[did], partitions[rollid ^ 1][did]._binsts + pbcnt[did],
                sizeof(int), cudaMemcpyDefault, cuDev.stream_compute()));
            cuDev.syncStream<streamIdx::Compute>();
          }
          timer.tock(fmt::format("GPU[{}] frame {} step {} update_partition",
                                 did, curFrame, curStep));

          timer.tick();
          cuDev.compute_launch({(pbcnt[did] + 127) / 128, 128},
                               register_neighbor_blocks, (uint32_t)pbcnt[did],
                               partitions[rollid ^ 1][did]);
          auto prev_nbcnt = nbcnt[did];
          checkCudaErrors(cudaMemcpyAsync(
              &nbcnt[did], partitions[rollid ^ 1][did]._cnt, sizeof(int),
              cudaMemcpyDefault, cuDev.stream_compute()));
          cuDev.compute_launch({(pbcnt[did] + 127) / 128, 128},
                               register_exterior_blocks, (uint32_t)pbcnt[did],
                               partitions[rollid ^ 1][did]);
          checkCudaErrors(cudaMemcpyAsync(
              &ebcnt[did], partitions[rollid ^ 1][did]._cnt, sizeof(int),
              cudaMemcpyDefault, cuDev.stream_compute()));
          cuDev.syncStream<streamIdx::Compute>();
          timer.tock(fmt::format("GPU[{}] frame {} step {} build_partition",
                                 did, curFrame, curStep));
          fmt::print(fmt::emphasis::bold | fg(fmt::color::yellow),
                     "block count on device {}: {}, {}, {} [{}]; {} [{}]\n",
                     did, pbcnt[did], nbcnt[did], ebcnt[did],
                     curNumActiveBlocks[did], bincnt[did],
                     curNumActiveBins[did]);

          /// check capacity
          if (checkedCnts[did][0] > 0) {
            gridBlocks[0][did].resize(device_allocator{},
                                      curNumActiveBlocks[did]);
          }
          timer.tick();
          gridBlocks[0][did].reset(ebcnt[did], cuDev);
          cuDev.compute_launch(
              {prev_nbcnt, g_blockvolume}, copy_selected_grid_blocks,
              (const ivec3 *)partitions[rollid][did]._activeKeys,
              partitions[rollid ^ 1][did], (const int *)activeBlockMarks,
              gridBlocks[1][did], gridBlocks[0][did]);
          cuDev.syncStream<streamIdx::Compute>();
          timer.tock(fmt::format("GPU[{}] frame {} step {} copy_blocks", did,
                                 curFrame, curStep));
          /// check capacity
          if (checkedCnts[did][0] > 0) {
            gridBlocks[1][did].resize(device_allocator{},
                                      curNumActiveBlocks[did]);
            tmps[did].resize(curNumActiveBlocks[did]);
          }
        });
        sync();
        rollid ^= 1;
        dt = nextDt;
      }
      issue([this](int did) {
        IO::flush();
        output_model(did);
      });
      sync();
      nextTime = 1.f * (curFrame + 1) / fps;
      fmt::print(fmt::emphasis::bold | fg(fmt::color::red),
                 "-----------------------------------------------------------"
                 "-----\n");
    }
  }
  void output_model(int did) {
    auto &cuDev = Cuda::ref_cuda_context(did);
    cuDev.setContext();
    CudaTimer timer{cuDev.stream_compute()};
    timer.tick();
    int parcnt, *d_parcnt = (int *)cuDev.borrow(sizeof(int));
    checkCudaErrors(
        cudaMemsetAsync(d_parcnt, 0, sizeof(int), cuDev.stream_compute()));
    match(particleBins[rollid][did])([&](const auto &pb) {
      cuDev.compute_launch({pbcnt[did], 128}, retrieve_particle_buffer,
                           partitions[rollid][did], partitions[rollid ^ 1][did],
                           pb, particles[did], d_parcnt);
    });
    checkCudaErrors(cudaMemcpyAsync(&parcnt, d_parcnt, sizeof(int),
                                    cudaMemcpyDefault, cuDev.stream_compute()));
    cuDev.syncStream<streamIdx::Compute>();
    fmt::print(fg(fmt::color::red), "total number of particles {}\n", parcnt);
    models[did].resize(parcnt);
    checkCudaErrors(cudaMemcpyAsync(models[did].data(),
                                    (void *)&particles[did].val_1d(_0, 0),
                                    sizeof(std::array<float, 3>) * (parcnt),
                                    cudaMemcpyDefault, cuDev.stream_compute()));
    cuDev.syncStream<streamIdx::Compute>();
    std::string fn = std::string{"model"} + "_dev[" + std::to_string(did) +
                     "]_frame[" + std::to_string(curFrame) + "].bgeo";
    IO::insert_job(
        [fn, model = models[did]]() { write_partio<float, 3>(fn, model); });
    timer.tock(fmt::format("GPU[{}] frame {} step {} retrieve_particles", did,
                           curFrame, curStep));
  }
  void initial_setup(int did) {
    auto &cuDev = Cuda::ref_cuda_context(did);
    cuDev.setContext();
    CudaTimer timer{cuDev.stream_compute()};
    timer.tick();
    cuDev.compute_launch({(pcnt[did] + 255) / 256, 256}, activate_blocks,
                         pcnt[did], particles[did], partitions[rollid][did]);
    checkCudaErrors(cudaMemcpyAsync(&pbcnt[did], partitions[rollid][did]._cnt,
                                    sizeof(int), cudaMemcpyDefault,
                                    cuDev.stream_compute()));
    timer.tock(fmt::format("GPU[{}] step {} init_table", did, curStep));

    timer.tick();
    cuDev.resetMem();
    // particle block
    cuDev.compute_launch({(pcnt[did] + 255) / 256, 256},
                         build_particle_cell_buckets, pcnt[did], particles[did],
                         partitions[rollid][did]);
    // bucket, binsts
    cuDev.syncStream<streamIdx::Compute>();
    partitions[rollid][did].buildParticleBuckets(cuDev, pbcnt[did]);
    {
      int *binpbs = tmps[did].binpbs;
      cuDev.compute_launch({(pbcnt[did] + 1 + 127) / 128, 128},
                           compute_bin_capacity, pbcnt[did] + 1,
                           (const int *)partitions[rollid][did]._ppbs, binpbs);
      exclScan(pbcnt[did] + 1, binpbs, partitions[rollid][did]._binsts, cuDev);
      checkCudaErrors(cudaMemcpyAsync(
          &bincnt[did], partitions[rollid][did]._binsts + pbcnt[did],
          sizeof(int), cudaMemcpyDefault, cuDev.stream_compute()));
      cuDev.syncStream<streamIdx::Compute>();
    }
    match(particleBins[rollid][did])([&](const auto &pb) {
      cuDev.compute_launch({pbcnt[did], 128}, array_to_buffer, particles[did],
                           pb, partitions[rollid][did]);
    });
    // grid block
    cuDev.compute_launch({(pbcnt[did] + 127) / 128, 128},
                         register_neighbor_blocks, (uint32_t)pbcnt[did],
                         partitions[rollid][did]);
    checkCudaErrors(cudaMemcpyAsync(&nbcnt[did], partitions[rollid][did]._cnt,
                                    sizeof(int), cudaMemcpyDefault,
                                    cuDev.stream_compute()));
    cuDev.compute_launch({(pbcnt[did] + 127) / 128, 128},
                         register_exterior_blocks, (uint32_t)pbcnt[did],
                         partitions[rollid][did]);
    checkCudaErrors(cudaMemcpyAsync(&ebcnt[did], partitions[rollid][did]._cnt,
                                    sizeof(int), cudaMemcpyDefault,
                                    cuDev.stream_compute()));
    cuDev.syncStream<streamIdx::Compute>();
    partitions[rollid][did].copy_to(partitions[rollid ^ 1][did], ebcnt[did],
                                    cuDev.stream_compute());
    timer.tock(fmt::format("GPU[{}] step {} init_partition", did, curStep));
    // not necessary
    checkCudaErrors(cudaMemcpyAsync(partitions[rollid ^ 1][did]._activeKeys,
                                    partitions[rollid][did]._activeKeys,
                                    sizeof(ivec3) * ebcnt[did],
                                    cudaMemcpyDefault, cuDev.stream_compute()));
    fmt::print(fmt::emphasis::bold | fg(fmt::color::yellow),
               "block count on device {}: {}, {}, {} [{}]; {} [{}]\n", did,
               pbcnt[did], nbcnt[did], ebcnt[did], curNumActiveBlocks[did],
               bincnt[did], curNumActiveBins[did]);

    timer.tick();
    gridBlocks[0][did].reset(nbcnt[did], cuDev);
    cuDev.compute_launch({(pcnt[did] + 255) / 256, 256}, rasterize, pcnt[did],
                         particles[did], gridBlocks[0][did],
                         partitions[rollid][did], dt, getMass(did));
    cuDev.compute_launch({pbcnt[did], 128}, init_adv_bucket,
                         (const int *)partitions[rollid][did]._ppbs,
                         partitions[rollid][did]._blockbuckets);
    cuDev.syncStream<streamIdx::Compute>();
    timer.tock(fmt::format("GPU[{}] step {} init_grid", did, curStep));
  }

  ///
  /// animation runtime settings
  float dt, nextDt, dtDefault, curTime, maxVel;
  uint64_t curFrame, curStep, fps;
  /// data on device, double buffering
  std::vector<GridBuffer> gridBlocks[2];
  std::vector<particle_buffer_t> particleBins[2];
  std::vector<Partition<0>> partitions[2];
  vec<ParticleArray, config::g_device_cnt> particles;
  vec<float *, config::g_device_cnt> d_maxVels;
  struct {
    void *base;
    float *d_maxVel;
    int *d_tmp;
    int *activeBlockMarks;
    int *destinations;
    int *sources;
    int *binpbs;
    void alloc(int maxBlockCnt) {
      checkCudaErrors(cudaMalloc(&base, sizeof(int) * (maxBlockCnt * 5 + 1)));
      d_maxVel = (float *)((char *)base + sizeof(int) * maxBlockCnt * 5);
      d_tmp = (int *)((uintptr_t)base);
      activeBlockMarks = (int *)((char *)base + sizeof(int) * maxBlockCnt);
      destinations = (int *)((char *)base + sizeof(int) * maxBlockCnt * 2);
      sources = (int *)((char *)base + sizeof(int) * maxBlockCnt * 3);
      binpbs = (int *)((char *)base + sizeof(int) * maxBlockCnt * 4);
    }
    void dealloc() {
      cudaDeviceSynchronize();
      checkCudaErrors(cudaFree(base));
    }
    void resize(int maxBlockCnt) {
      dealloc();
      alloc(maxBlockCnt);
    }
  } tmps[config::g_device_cnt];

  /// data on host
  static_assert(std::is_same<GridBufferDomain::index_type, int>::value,
                "block index type is not int");
  char rollid;
  std::size_t curNumActiveBlocks[config::g_device_cnt],
      curNumActiveBins[config::g_device_cnt],
      checkedCnts[config::g_device_cnt][2];
  vec<float, config::g_device_cnt> maxVels;
  vec<int, config::g_device_cnt> pbcnt, nbcnt, ebcnt, bincnt; ///< num blocks
  vec<uint32_t, config::g_device_cnt> pcnt;                   ///< num particles
  // std::vector<stack_allocator<device_memory_resource>> mempools;
  std::vector<float> durations[config::g_device_cnt + 1];
  std::vector<std::array<float, 3>> models[config::g_device_cnt];

  /// control
  bool bRunning;
  threadsafe_queue<std::function<void(int)>> jobs[config::g_device_cnt];
  std::thread ths[config::g_device_cnt]; ///< thread is not trivial
  std::mutex mut_slave, mut_ctrl;
  std::condition_variable cv_slave, cv_ctrl;
  std::atomic_uint idleCnt{0};

  /// computations per substep
  std::vector<std::function<void(int)>> init_tasks;
  std::vector<std::function<void(int)>> loop_tasks;
};

} // namespace mn

#endif