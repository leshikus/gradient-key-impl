// gradient_complement: exact t_{n,m}(G) near the complete cube, m = 2^n - r, by enumerating the r missing words.
//
// A table with distinct columns is an ordered choice of m distinct words of {0,1}^n, and the average leaf depth
// Lbar of G does not depend on the order of the columns. So
//   t_{n,m}(G) = (sum over m-subsets X of {0,1}^n of the depth sum of G(X)) / (m * C(2^n, m)),
// and X is the complement of an r-subset R. Complementing row i (all words w -> w ^ e_i) keeps every quality and
// only swaps the children of the forks that test row i, so the depth sum is invariant under XOR translations
// w -> w ^ v. Hence, with f(R) the depth sum of G({0,1}^n minus R),
//   sum_R f(R) = (2^n / r) * sum_{R containing word 0} f(R),
// and t_{n,m}(G) = sum_{R containing 0} f(R) / (m * C(2^n - 1, r - 1)).
// The engine enumerates the C(2^n - 1, r - 1) sets R containing word 0 and builds G on the m remaining words
// as 64-bit masks (n <= 6). It prints "n,m,numerator,denominator" (not reduced) with
// numerator = sum of f(R) and denominator = m * (number of sets R enumerated).
//
// Usage: gradient_complement n r threads     (2 <= n <= 6, 1 <= r <= 2^n - 2)
//
// SPDX-License-Identifier: MIT
#include <atomic>
#include <bit>
#include <charconv>
#include <cstdint>
#include <cstring>
#include <optional>
#include <print>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {

using u64 = std::uint64_t;
using u128 = unsigned __int128;

int N = 0;
u64 ROW[6];  // ROW[i]: the words with bit i equal to 1

// Sum of leaf depths of G on the word set S (distinct words, so a splitting row always exists).
u64 depth_sum(u64 S) {
  const int s = std::popcount(S);
  if (s <= 1) return 0;
  int best = -1, bq = 0;
  for (int i = 0; i < N; ++i) {
    const int k = std::popcount(S & ROW[i]);
    const int q = k < s - k ? k : s - k;
    if (q > bq) {
      bq = q;
      best = i;
    }
  }
  const u64 one = S & ROW[best];
  return static_cast<u64>(s) + depth_sum(one) + depth_sum(S & ~one);
}

std::string str(u128 v) {
  if (v == 0) return "0";
  std::string s;
  while (v) {
    s.push_back(static_cast<char>('0' + static_cast<int>(v % 10)));
    v /= 10;
  }
  return {s.rbegin(), s.rend()};
}

std::optional<int> parse_int(const char* s) {
  int v = 0;
  const char* end = s + std::strlen(s);
  auto [p, ec] = std::from_chars(s, end, v);
  if (ec != std::errc{} || p != end) return std::nullopt;
  return v;
}

}  // namespace

int main(int argc, char** argv) {
  std::optional<int> n, r, threads;
  if (argc == 4) n = parse_int(argv[1]), r = parse_int(argv[2]), threads = parse_int(argv[3]);
  if (!n || !r || !threads || *n < 2 || *n > 6 || *r < 1 || *r > (1 << *n) - 2 || *threads < 1) {
    std::println(stderr, "usage: gradient_complement n r threads   (2 <= n <= 6, 1 <= r <= 2^n - 2)");
    return 2;
  }
  N = *n;
  const int W = 1 << N, R = *r, T = *threads;
  const u64 FULL = W == 64 ? ~u64{0} : (u64{1} << W) - 1;
  for (int i = 0; i < N; ++i) {
    ROW[i] = 0;
    for (int w = 0; w < W; ++w)
      if ((w >> i) & 1) ROW[i] |= u64{1} << w;
  }
  // R contains word 0. Tasks: the next one or two words of R, for load balance.
  std::vector<std::pair<int, int>> tasks;
  if (R == 1)
    tasks.emplace_back(0, 0);
  else if (R == 2)
    for (int a = 1; a < W; ++a) tasks.emplace_back(a, 0);
  else
    for (int a = 1; a < W; ++a)
      for (int b = a + 1; b < W; ++b) tasks.emplace_back(a, b);
  std::atomic<std::size_t> next{0};
  std::vector<u128> sum(static_cast<std::size_t>(T), 0), count(static_cast<std::size_t>(T), 0);
  {
    std::vector<std::jthread> pool;
    for (int t = 0; t < T; ++t)
      pool.emplace_back([&, t] {
        u128& S = sum[static_cast<std::size_t>(t)];
        u128& C = count[static_cast<std::size_t>(t)];
        for (std::size_t ti = next++; ti < tasks.size(); ti = next++) {
          const auto [a, b] = tasks[ti];
          u64 base = 1;  // word 0
          int from = W, left = 0;
          if (R == 2) {
            base |= u64{1} << a;
          } else if (R >= 3) {
            base |= (u64{1} << a) | (u64{1} << b);
            from = b + 1;
            left = R - 3;
          }
          auto rec = [&](auto&& self, int pos, int start, u64 missing) -> void {
            if (pos == left) {
              S += depth_sum(FULL & ~missing);
              ++C;
              return;
            }
            for (int w = start; w <= W - (left - pos); ++w) self(self, pos + 1, w + 1, missing | (u64{1} << w));
          };
          rec(rec, 0, from, base);
        }
      });
  }
  u128 S = 0, C = 0;
  for (int t = 0; t < T; ++t) S += sum[static_cast<std::size_t>(t)], C += count[static_cast<std::size_t>(t)];
  const int m = W - R;
  std::println("{},{},{},{}", N, m, str(S), str(C * static_cast<u128>(m)));
  return 0;
}
