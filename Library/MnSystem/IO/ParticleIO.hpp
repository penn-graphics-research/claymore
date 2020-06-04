#ifndef __PARTICLE_IO_HPP_
#define __PARTICLE_IO_HPP_
#include "PoissonDisk/SampleGenerator.h"
#include <MnBase/Math/Vec.h>
#include <Partio.h>
#include <array>
#include <string>
#include <vector>

namespace mn {

template <typename T, std::size_t dim>
void write_partio(std::string filename,
                  const std::vector<std::array<T, dim>> &data,
                  std::string tag = std::string{"position"}) {
  Partio::ParticlesDataMutable *parts = Partio::create();

  Partio::ParticleAttribute attrib =
      parts->addAttribute(tag.c_str(), Partio::VECTOR, dim);

  parts->addParticles(data.size());
  for (int idx = 0; idx < (int)data.size(); ++idx) {
    float *val = parts->dataWrite<float>(attrib, idx);
    for (int k = 0; k < dim; k++)
      val[k] = data[idx][k];
  }
  Partio::write(filename.c_str(), *parts);
  parts->release();
}

auto read_sdf(std::string fn, float ppc, float dx, vec<float, 3> offset,
              vec<float, 3> lengths) {
  std::vector<std::array<float, 3>> data;
  std::string fileName = std::string(AssetDirPath) + "MpmParticles/" + fn;

  float levelsetDx;
  SampleGenerator pd;
  std::vector<float> samples;
  vec<float, 3> mins, maxs, scales;
  vec<int, 3> maxns;
  pd.LoadSDF(fileName, levelsetDx, mins[0], mins[1], mins[2], maxns[0],
             maxns[1], maxns[2]);
  maxs = maxns.cast<float>() * levelsetDx;
  scales = lengths / (maxs - mins);
  float scale = scales[0] < scales[1] ? scales[0] : scales[1];
  scale = scales[2] < scale ? scales[2] : scale;

  float samplePerLevelsetCell = ppc * levelsetDx / dx * scale;

  pd.GenerateUniformSamples(samplePerLevelsetCell, samples);

  for (int i = 0, size = samples.size() / 3; i < size; i++) {
    vec<float, 3> p{samples[i * 3 + 0], samples[i * 3 + 1], samples[i * 3 + 2]};
    p = (p - mins) * scale + offset;
    // particle[0] = ((samples[i * 3 + 0]) + offset[0]);
    // particle[1] = ((samples[i * 3 + 1]) + offset[1]);
    // particle[2] = ((samples[i * 3 + 2]) + offset[2]);
    data.push_back(std::array<float, 3>{p[0], p[1], p[2]});
  }
  printf("[%f, %f, %f] - [%f, %f, %f], scale %f, parcnt %d, lsdx %f, dx %f\n",
         mins[0], mins[1], mins[2], maxs[0], maxs[1], maxs[2], scale,
         (int)data.size(), levelsetDx, dx);
  return data;
}

} // namespace mn

#endif