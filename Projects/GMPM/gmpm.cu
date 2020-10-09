#include "gmpm_simulator.cuh"
#include <MnBase/Geometry/GeometrySampler.h>
#include <MnBase/Math/Vec.h>
#include <MnSystem/Cuda/Cuda.h>
#include <MnSystem/IO/IO.h>
#include <MnSystem/IO/ParticleIO.hpp>

#include <cxxopts.hpp>
#include <fmt/color.h>
#include <fmt/core.h>
#include <fstream>

#if 0
#include <ghc/filesystem.hpp>
namespace fs = ghc::filesystem;
#else
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#endif

#include <rapidjson/document.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>
namespace rj = rapidjson;
static const char *kTypeNames[] = {"Null",  "False",  "True",  "Object",
                                   "Array", "String", "Number"};

// dragon_particles.bin, 775196
// cube256_2.bin, 1048576
// two_dragons.bin, 388950

decltype(auto) load_model(std::size_t pcnt, std::string filename) {
  std::vector<std::array<float, 3>> rawpos(pcnt);
  auto addr_str = std::string(AssetDirPath) + "MpmParticles/";
  auto f = fopen((addr_str + filename).c_str(), "rb");
  std::fread((float *)rawpos.data(), sizeof(float), rawpos.size() * 3, f);
  std::fclose(f);
  return rawpos;
}

struct SimulatorConfigs {
  int _dim;
  float _dx, _dxInv;
  int _resolution;
  float _gravity;
  std::vector<float> _offset;
} simConfigs;

void parse_scene(std::string fn,
                 std::unique_ptr<mn::GmpmSimulator> &benchmark) {

  fs::path p{fn};
  if (p.empty())
    fmt::print("file not exist {}\n", fn);
  else {
    std::size_t size = fs::file_size(p);
    std::string configs;
    configs.resize(size);

    std::ifstream istrm(fn);
    if (!istrm.is_open())
      fmt::print("cannot open file {}\n", fn);
    else
      istrm.read(const_cast<char *>(configs.data()), configs.size());
    istrm.close();
    fmt::print("load the scene file of size {}\n", size);

    rj::Document doc;
    doc.Parse(configs.data());
    for (rj::Value::ConstMemberIterator itr = doc.MemberBegin();
         itr != doc.MemberEnd(); ++itr) {
      fmt::print("Scene member {} is {}\n", itr->name.GetString(),
                 kTypeNames[itr->value.GetType()]);
    }
    {
      auto it = doc.FindMember("simulation");
      if (it != doc.MemberEnd()) {
        auto &sim = it->value;
        if (sim.IsObject()) {
          fmt::print(
              fg(fmt::color::cyan),
              "simulation: gpuid[{}], defaultDt[{}], fps[{}], frames[{}]\n",
              sim["gpu"].GetInt(), sim["default_dt"].GetFloat(),
              sim["fps"].GetInt(), sim["frames"].GetInt());
          benchmark = std::make_unique<mn::GmpmSimulator>(
              sim["gpuid"].GetInt(), sim["default_dt"].GetFloat(),
              sim["fps"].GetInt(), sim["frames"].GetInt());
        }
      }
    } ///< end simulation parsing
    {
      auto it = doc.FindMember("models");
      if (it != doc.MemberEnd()) {
        if (it->value.IsArray()) {
          fmt::print("has {} models\n", it->value.Size());
          for (auto &model : it->value.GetArray()) {
            std::string constitutive{model["constitutive"].GetString()};
            fmt::print(fg(fmt::color::green),
                       "model constitutive[{}], file[{}]\n", constitutive,
                       model["file"].GetString());
            fs::path p{model["file"].GetString()};

            auto initModel = [&](auto &positions, auto &velocity) {
              if (constitutive == "fixed_corotated") {
                benchmark->initModel<mn::material_e::FixedCorotated>(positions,
                                                                     velocity);
                benchmark->updateFRParameters(
                    model["rho"].GetFloat(), model["volume"].GetFloat(),
                    model["youngs_modulus"].GetFloat(),
                    model["poisson_ratio"].GetFloat());
              } else if (constitutive == "jfluid") {
                benchmark->initModel<mn::material_e::JFluid>(positions,
                                                             velocity);
                benchmark->updateJFluidParameters(
                    model["rho"].GetFloat(), model["volume"].GetFloat(),
                    model["bulk_modulus"].GetFloat(), model["gamma"].GetFloat(),
                    model["viscosity"].GetFloat());
              } else if (constitutive == "nacc") {
                benchmark->initModel<mn::material_e::NACC>(positions, velocity);
                benchmark->updateNACCParameters(
                    model["rho"].GetFloat(), model["volume"].GetFloat(),
                    model["youngs_modulus"].GetFloat(),
                    model["poisson_ratio"].GetFloat(), model["beta"].GetFloat(),
                    model["xi"].GetFloat());
              } else if (constitutive == "sand") {
                benchmark->initModel<mn::material_e::Sand>(positions, velocity);
              }
            };
            mn::vec<float, 3> offset, span, velocity;
            for (int d = 0; d < 3; ++d)
              offset[d] = model["offset"].GetArray()[d].GetFloat(),
              span[d] = model["span"].GetArray()[d].GetFloat(),
              velocity[d] = model["velocity"].GetArray()[d].GetFloat();
            if (p.extension() == ".sdf") {
              auto positions = mn::read_sdf(
                  model["file"].GetString(), model["ppc"].GetFloat(),
                  mn::config::g_dx, mn::config::g_domain_size, offset, span);
              mn::IO::insert_job([&]() {
                mn::write_partio<float, 3>(std::string{p.stem()} + ".bgeo",
                                           positions);
              });
              mn::IO::flush();
              initModel(positions, velocity);
            }
          }
        }
      }
    } ///< end models parsing
  }
}

int main(int argc, char *argv[]) {
  using namespace mn;
  using namespace config;
  Cuda::startup();

  cxxopts::Options options("Scene_Loader", "Read simulation scene");
  options.add_options()(
      "f,file", "Scene Configuration File",
      cxxopts::value<std::string>()->default_value("scene.json"));
  auto results = options.parse(argc, argv);
  auto fn = results["file"].as<std::string>();
  fmt::print("loading scene [{}]\n", fn);

  std::unique_ptr<mn::GmpmSimulator> benchmark;
  parse_scene(fn, benchmark);
  if (false) {
    benchmark = std::make_unique<mn::GmpmSimulator>(1, 1e-4, 24, 60);

    constexpr auto LEN = 46;
    constexpr auto STRIDE = 56;
    constexpr auto MODEL_CNT = 1;
    for (int did = 0; did < 2; ++did) {
      std::vector<std::array<float, 3>> model;
      for (int i = 0; i < MODEL_CNT; ++i) {
        auto idx = (did * MODEL_CNT + i);
        model = sample_uniform_box(
            g_dx, ivec3{18 + (idx & 1 ? STRIDE : 0), 18, 18},
            ivec3{18 + (idx & 1 ? STRIDE : 0) + LEN, 18 + LEN, 18 + LEN});
      }
      benchmark->initModel<mn::material_e::FixedCorotated>(
          model, vec<float, 3>{0.f, 0.f, 0.f});
    }
  }
  getchar();

  benchmark->main_loop();
  ///
  IO::flush();
  benchmark.reset();
  ///
  Cuda::shutdown();
  return 0;
}