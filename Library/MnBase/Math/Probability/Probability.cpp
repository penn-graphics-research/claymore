#include "Probability.h"

#ifdef _WIN32
#	ifndef M_PI
#		define M_PI 3.14159265358979323846//NOLINT(cppcoreguidelines-macro-usage) M_PI is usually a macro
#	endif
#endif

namespace mn {
Propability::Propability()
	: distribution(0.0, 1.0)
	, generator(device()) {}

double Propability::pdf(double lambda, int k) {
	double pdf_value = 1;
	int i;
	for(i = 1; i <= k; ++i) {
		pdf_value *= lambda / static_cast<double>(i);
	}
	return pdf_value * exp(-1.0 * lambda);
}

double Propability::pdf(double u, double o, double x) {
	static const double CO = 1.0 / sqrt(2 * M_PI);			  //NOLINT(readability-magic-numbers) Formula specific
	double index		   = -(x - u) * (x - u) / 2.0 / o / o;//NOLINT(readability-magic-numbers) Formula specific
	return CO / o * exp(index);
}

double Propability::anti_normal_pdf(double u, double o, double x) {
	static const double CO = 1.0 / sqrt(2 * M_PI);			  //NOLINT(readability-magic-numbers) Formula specific
	double index		   = -(x - u) * (x - u) / 2.0 / o / o;//NOLINT(readability-magic-numbers) Formula specific
	return 1 - CO / o * exp(index);
}

int Propability::rand_p(double lambda) {
	double u   = distribution(generator);
	int x	   = 0;
	double cdf = exp(-1.0 * lambda);
	while(u >= cdf) {
		x++;
		cdf += pdf(lambda, x);
	}
	return x;
}

int Propability::rand_normal(double u, double o) {
	double val = distribution(generator);
	int x	   = 0;
	double cdf = 0;// pdf(u, o, x)
	while(val >= cdf) {
		x++;
		cdf += pdf(u, o, static_cast<double>(x));
	}
	return x;
}

int Propability::rand_anti_normal(double u, double o) {
	double val = distribution(generator);
	int x	   = 0;
	double cdf = 0;// pdf(u, o, x)
	while(val >= cdf) {
		x++;
		cdf += anti_normal_pdf(u, o, static_cast<double>(x));
	}
	return x;
}
}// namespace mn