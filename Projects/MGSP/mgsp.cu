#include <MnBase/Geometry/GeometrySampler.h>
#include <MnBase/Math/Vec.h>
#include <MnSystem/Cuda/Cuda.h>
#include <MnSystem/IO/IO.h>

#include <MnSystem/IO/ParticleIO.hpp>

#include "mgsp_benchmark.cuh"
#include "partition_domain.h"

// dragon_particles.bin, 775196
// cube256_2.bin, 1048576
// two_dragons.bin, 388950

constexpr size_t DRAGON_PARTICLES_SIZE = 775196;
constexpr size_t CUBE_256_2_SIZE	   = 1048576;
constexpr size_t TWO_DRAGONS_SIZE	   = 388950;

constexpr size_t SCENARIO = 3;

decltype(auto) load_model(std::size_t particle_counts, const std::string& filename) {
	std::vector<std::array<float, mn::config::NUM_DIMENSIONS>> rawpos(particle_counts);
	auto addr_str = std::string(AssetDirPath) + "MpmParticles/";
	FILE* f;
	fopen_s(&f, (addr_str + filename).c_str(), "rb");
	std::fread(rawpos.data(), sizeof(float), rawpos.size() * mn::config::NUM_DIMENSIONS, f);
	std::fclose(f);
	return rawpos;
}

//NOLINTBEGIN(readability-magic-numbers) Numbers are scenario parameters
// load from analytic levelset
// init models
void init_models(std::array<std::vector<std::array<float, mn::config::NUM_DIMENSIONS>>, mn::config::G_DEVICE_COUNT> models, int opt = 0) {
	(void) DRAGON_PARTICLES_SIZE;
	(void) CUBE_256_2_SIZE;
	(void) TWO_DRAGONS_SIZE;
	switch(opt) {
		case 0:
			models[0] = load_model(DRAGON_PARTICLES_SIZE, "dragon_particles.bin");
			models[1] = read_sdf(std::string {"two_dragons.sdf"}, 8.f, mn::config::G_DX, mn::vec<float, mn::config::NUM_DIMENSIONS> {0.5f, 0.5f, 0.5f}, mn::vec<float, mn::config::NUM_DIMENSIONS> {0.2f, 0.2f, 0.2f});
			break;
		case 1:
			models[0] = load_model(DRAGON_PARTICLES_SIZE, "dragon_particles.bin");
			models[1] = load_model(DRAGON_PARTICLES_SIZE, "dragon_particles.bin");
			for(auto& pt: models[1]) {
				pt[1] -= 0.3f;
			}
			break;
		case 2: {
			constexpr auto LEN		   = 54;
			constexpr auto STRIDE	   = 56;
			constexpr auto MODEL_COUNT = 1;
			for(int did = 0; did < mn::config::G_DEVICE_COUNT; ++did) {
				models[did].clear();
				std::vector<std::array<float, mn::config::NUM_DIMENSIONS>> model;
				for(int i = 0; i < MODEL_COUNT; ++i) {
					auto idx = (did * MODEL_COUNT + i);
					model	 = sample_uniform_box(mn::config::G_DX, mn::ivec3 {18 + ((idx & 1) != 0 ? STRIDE : 0), 18, 18}, mn::ivec3 {18 + ((idx & 1) != 0 ? STRIDE : 0) + LEN, 18 + LEN, 18 + LEN});
					models[did].insert(models[did].end(), model.begin(), model.end());
				}
			}
		} break;
		case 3: {
			constexpr auto LEN		   = 72;// 54;
			constexpr auto STRIDE	   = (mn::config::G_DOMAIN_SIZE / 2);
			constexpr auto MODEL_COUNT = 1;
			for(int did = 0; did < mn::config::G_DEVICE_COUNT; ++did) {
				models[did].clear();
				std::vector<std::array<float, mn::config::NUM_DIMENSIONS>> model;
				for(int i = 0; i < MODEL_COUNT; ++i) {
					auto idx = (did * MODEL_COUNT + i);
					model	 = sample_uniform_box(mn::config::G_DX, mn::ivec3 {18 + ((idx & 1) != 0 ? STRIDE : 0), 18, 18 + ((idx & 2) != 0 ? STRIDE : 0)}, mn::ivec3 {18 + ((idx & 1) != 0 ? STRIDE : 0) + LEN, 18 + LEN / 3, 18 + ((idx & 2) != 0 ? STRIDE : 0) + LEN});
					models[did].insert(models[did].end(), model.begin(), model.end());
				}
			}
		} break;
		default:
			break;
	}
}
//NOLINTEND(readability-magic-numbers)

int main() {
	mn::Cuda::startup();

	std::array<std::vector<std::array<float, mn::config::NUM_DIMENSIONS>>, mn::config::G_DEVICE_COUNT> models;

	auto benchmark = std::make_unique<mn::MgspBenchmark>();
	/// init
	init_models(models, SCENARIO);

	for(int did = 0; did < mn::config::G_DEVICE_COUNT; ++did) {
		benchmark->init_model(did, models[did]);
	}
	// benchmark->init_boundary("candy_base");

	benchmark->main_loop();
	///
	mn::IO::flush();
	benchmark.reset();
	///
	mn::Cuda::shutdown();
	return 0;
}