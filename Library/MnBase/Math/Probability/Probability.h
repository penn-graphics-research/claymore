#ifndef PROBABILITY_H
#define PROBABILITY_H

#include <functional>
#include <random>

namespace mn {

class Propability {
	std::uniform_real_distribution<double> distribution;
	std::random_device device;
	std::mt19937 generator;

	Propability();

   public:
	static double pdf(double lambda, int k);
	int rand_p(double lambda);
	static double anti_normal_pdf(double u, double o, double x);

	static double pdf(double u, double o, double x);
	int rand_normal(double u, double o);
	int rand_anti_normal(double u, double o);
};
}// namespace mn

#endif