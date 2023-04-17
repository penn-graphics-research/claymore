#include <MnBase/Geometry/GeometrySampler.h>
#include <MnBase/Math/Vec.h>
#include <MnSystem/Cuda/Cuda.h>
#include <MnSystem/IO/IO.h>
#include <fmt/color.h>
#include <fmt/core.h>

#include <MnSystem/IO/ParticleIO.hpp>
#include <cxxopts.hpp>
#include <filesystem>
#include <fstream>

#include "gmpm_simulator.cuh"
namespace fs = std::filesystem;

#include <rapidjson/document.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>
namespace rj = rapidjson;
namespace {
const std::array K_TYPE_NAMES {"Null", "False", "True", "Object", "Array", "String", "Number"};
}// namespace

// dragon_particles.bin, 775196
// cube256_2.bin, 1048576
// two_dragons.bin, 388950

decltype(auto) load_model(std::size_t pcnt, const std::string& filename) {
	std::vector<std::array<float, mn::config::NUM_DIMENSIONS>> rawpos(pcnt);
	auto addr_str = std::string(AssetDirPath) + "MpmParticles/";
	FILE* f;
	fopen_s(&f, (addr_str + filename).c_str(), "rb");
	std::fread(rawpos.data(), sizeof(float), rawpos.size() * mn::config::NUM_DIMENSIONS, f);
	std::fclose(f);
	return rawpos;
}

struct SimulatorConfigs {
	int dim					  = 0;
	float dx				  = NAN;
	float dx_inv			  = NAN;
	int resolution			  = 0;
	float gravity			  = NAN;
	std::vector<float> offset = {};
} const SIM_CONFIGS;

namespace {
template<typename T>
inline bool check_member(const T& model, const char* member) {
	if(!model.HasMember(member)) {
		fmt::print("Membert not found: {}\n", member);
		return false;
	} else {
		return true;
	}
}
}// namespace

//NOLINTBEGIN(clang-analyzer-cplusplus.PlacementNew) check_member prevents the error case
void parse_scene(const std::string& fn, std::unique_ptr<mn::GmpmSimulator>& benchmark) {
	fs::path p {fn};
	if(!p.is_absolute()) {
		p = fs::relative(p);
	}
	if(!fs::exists(p)) {
		fmt::print("file not exist {}\n", fn);
	} else {
		std::size_t size = fs::file_size(p);
		std::string configs;
		configs.resize(size);

		std::ifstream istrm(fn);
		if(!istrm.is_open()) {
			fmt::print("cannot open file {}\n", fn);
		} else {
			istrm.read(configs.data(), static_cast<std::streamsize>(configs.size()));
		}
		istrm.close();
		fmt::print("load the scene file of size {}\n", size);

		rj::Document doc;
		doc.Parse(configs.data());
		for(rj::Value::ConstMemberIterator itr = doc.MemberBegin(); itr != doc.MemberEnd(); ++itr) {
			fmt::print("Scene member {} is {}\n", itr->name.GetString(), K_TYPE_NAMES[itr->value.GetType()]);
		}
		{
			auto it = doc.FindMember("simulation");
			if(it != doc.MemberEnd()) {
				auto& sim = it->value;
				if(sim.IsObject()) {
					fmt::print(fg(fmt::color::cyan), "simulation: gpuid[{}], defaultDt[{}], fps[{}], frames[{}]\n", sim["gpuid"].GetInt(), sim["default_dt"].GetFloat(), sim["fps"].GetInt(), sim["frames"].GetInt());
					benchmark = std::make_unique<mn::GmpmSimulator>(sim["gpuid"].GetInt(), sim["default_dt"].GetFloat(), sim["fps"].GetInt(), sim["frames"].GetInt());
				}
			}
		}///< end simulation parsing
		{
			auto it = doc.FindMember("models");
			if(it != doc.MemberEnd()) {
				if(it->value.IsArray()) {
					fmt::print("has {} models\n", it->value.Size());
					for(auto& model: it->value.GetArray()) {
						if(!check_member(model, "constitutive") || !check_member(model, "file")) {
							return;
						}

						std::string constitutive {model["constitutive"].GetString()};

						fmt::print(fg(fmt::color::green), "model constitutive[{}], file[{}]\n", constitutive, model["file"].GetString());

						fs::path p {model["file"].GetString()};

						auto init_model = [&](auto& positions, auto& velocity) {
							if(constitutive == "fixed_corotated") {
								if(!check_member(model, "rho") || !check_member(model, "volume") || !check_member(model, "youngs_modulus") || !check_member(model, "poisson_ratio")) {
									return;
								}

								benchmark->init_model<mn::MaterialE::FIXED_COROTATED>(positions, velocity);
								benchmark->update_fr_parameters(model["rho"].GetFloat(), model["volume"].GetFloat(), model["youngs_modulus"].GetFloat(), model["poisson_ratio"].GetFloat());
							} else if(constitutive == "jfluid") {
								if(!check_member(model, "rho") || !check_member(model, "volume") || !check_member(model, "bulk_modulus") || !check_member(model, "gamma") || !check_member(model, "viscosity")) {
									return;
								}

								benchmark->init_model<mn::MaterialE::J_FLUID>(positions, velocity);
								benchmark->update_j_fluid_parameters(model["rho"].GetFloat(), model["volume"].GetFloat(), model["bulk_modulus"].GetFloat(), model["gamma"].GetFloat(), model["viscosity"].GetFloat());
							} else if(constitutive == "nacc") {
								if(!check_member(model, "rho") || !check_member(model, "volume") || !check_member(model, "youngs_modulus") || !check_member(model, "poisson_ratio") || !check_member(model, "beta") || !check_member(model, "xi")) {
									return;
								}

								benchmark->init_model<mn::MaterialE::NACC>(positions, velocity);
								benchmark->update_nacc_parameters(model["rho"].GetFloat(), model["volume"].GetFloat(), model["youngs_modulus"].GetFloat(), model["poisson_ratio"].GetFloat(), model["beta"].GetFloat(), model["xi"].GetFloat());
							} else if(constitutive == "sand") {
								benchmark->init_model<mn::MaterialE::SAND>(positions, velocity);
							} else {
								fmt::print("Unknown constitutive: {}", constitutive);
							}
						};
						mn::vec<float, mn::config::NUM_DIMENSIONS> offset;
						mn::vec<float, mn::config::NUM_DIMENSIONS> span;
						mn::vec<float, mn::config::NUM_DIMENSIONS> velocity;
						if(!check_member(model, "offset") || !check_member(model, "span") || !check_member(model, "velocity")) {
							return;
						}

						for(int d = 0; d < mn::config::NUM_DIMENSIONS; ++d) {
							offset[d] = model["offset"].GetArray()[d].GetFloat(), span[d] = model["span"].GetArray()[d].GetFloat(), velocity[d] = model["velocity"].GetArray()[d].GetFloat();
						}
						if(p.extension() == ".sdf") {
							if(!check_member(model, "ppc")) {
								return;
							}

							auto positions = mn::read_sdf(model["file"].GetString(), model["ppc"].GetFloat(), mn::config::G_DX, mn::config::G_DOMAIN_SIZE, offset, span);
							mn::IO::insert_job([&]() {
								mn::write_partio<float, mn::config::NUM_DIMENSIONS>(p.stem().string() + ".bgeo", positions);
							});
							mn::IO::flush();
							init_model(positions, velocity);
						}
					}
				}
			}
		}///< end models parsing
	}
}
//NOLINTEND(clang-analyzer-cplusplus.PlacementNew)

int main(int argc, char* argv[]) {
	mn::Cuda::startup();

	cxxopts::Options options("Scene_Loader", "Read simulation scene");
	options.add_options()("f,file", "Scene Configuration File", cxxopts::value<std::string>()->default_value("scene.json"));
	auto results = options.parse(argc, argv);
	auto fn		 = results["file"].as<std::string>();
	fmt::print("loading scene [{}]\n", fn);

	std::unique_ptr<mn::GmpmSimulator> benchmark;
	parse_scene(fn, benchmark);
	/*
	benchmark = std::make_unique<mn::GmpmSimulator>(1, 1e-4, 24, 60);

	constexpr auto LEN		 = 46;
	constexpr auto STRIDE	 = 56;
	constexpr auto MODEL_CNT = 3;
	for(int did = 0; did < 2; ++did) {
		std::vector<std::array<float, 3>> model;
		for(int i = 0; i < MODEL_CNT; ++i) {
			auto idx = (did * MODEL_CNT + i);
			model	 = sample_uniform_box(
				   gdx,
				   ivec3 {18 + (idx & 1 ? STRIDE : 0), 18, 18},
				   ivec3 {18 + (idx & 1 ? STRIDE : 0) + LEN, 18 + LEN, 18 + LEN}
			   );
		}
		benchmark->init_model<mn::MaterialE::FixedCorotated>(
			model,
			vec<float, 3> {0.f, 0.f, 0.f}
		);
	}
	*/
	getchar();

	benchmark->main_loop();
	///
	mn::IO::flush();
	benchmark.reset();
	///
	mn::Cuda::shutdown();
	return 0;
}