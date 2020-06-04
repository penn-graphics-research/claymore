#ifndef __BITS_H_
#define __BITS_H_
#include <bitset>
#include <limits>
#include <stdint.h>

namespace mn {
/**
 */
template <typename Integer>
constexpr Integer interleaved_bit_mask(int dim) noexcept {
  constexpr Integer unit{1};
  auto totalBits = sizeof(Integer) << 3;
  Integer mask = 0;
  for (decltype(totalBits) curBit = 0; curBit < totalBits; curBit += dim)
    mask |= (unit << curBit);
  return mask;
}
/**
 *	\fn uint32_t bit_length(uint32_t N)
 *	\brief compute the count of significant digits of a number
 *	\param N the number
 */
template <typename Integer> constexpr Integer bit_length(Integer N) noexcept {
  if (N > 0)
    return bit_length(N >> 1) + static_cast<Integer>(1);
  else
    return 0;
}
/**
 *	\fn uint32_t bit_count(uint32_t N)
 *	\brief compute the count of digits required to express integers in [0,
 *N) \param N the maximum of the range
 */
template <typename Integer> constexpr Integer bit_count(Integer N) noexcept {
  if (N > 0)
    return bit_length(N - 1);
  else
    return Integer{0};
}
/**
 *	\fn uint32_t next_power_of_two(uint32_t i)
 *	\brief compute the next power of two bigger than the number i
 *	\param i the number
 */
constexpr uint32_t next_power_of_two(uint32_t i) noexcept {
  i--;
  i |= i >> 1;
  i |= i >> 2;
  i |= i >> 4;
  i |= i >> 8;
  i |= i >> 16;
  return i + 1;
}

template <typename Integer>
constexpr Integer binary_reverse(Integer data,
                                 char loc = sizeof(Integer) * 8 - 1) {
  if (data == 0)
    return 0;
  return ((data & 1) << loc) | binary_reverse(data >> 1, loc - 1);
}

template <typename Integer>
constexpr unsigned count_leading_zeros(Integer data) {
  unsigned res{0};
  data = binary_reverse(data);
  if (data == 0)
    return sizeof(Integer) * 8;
  while ((data & 1) == 0)
    res++, data >>= 1;
  return res;
}

constexpr int bit_pack(const uint64_t mask, const uint64_t data) {
  uint64_t slresult = 0;
  uint64_t &ulresult{slresult};
  uint64_t uldata = data;
  int count = 0;
  ulresult = 0;

  uint64_t rmask = binary_reverse(mask);
  unsigned char lz{0};

  while (rmask) {
    lz = count_leading_zeros(rmask);
    uldata >>= lz;
    ulresult <<= 1;
    count++;
    ulresult |= (uldata & 1);
    uldata >>= 1;
    rmask <<= lz + 1;
  }
  ulresult <<=
      64 - count; // 64 means 64 bits ... maybe not use a constant 64 ...
  ulresult = binary_reverse(ulresult);
  return (int)slresult;
}

constexpr uint64_t bit_spread(const uint64_t mask, const int data) {
  uint64_t rmask = binary_reverse(mask);
  int dat = data;
  uint64_t result = 0;
  unsigned char lz{0};
  while (rmask) {
    lz = count_leading_zeros(rmask) + 1;
    result = result << lz | (dat & 1);
    dat >>= 1, rmask <<= lz;
  }
  result = binary_reverse(result) >> count_leading_zeros(mask);
  return result;
}
} // namespace mn

#endif