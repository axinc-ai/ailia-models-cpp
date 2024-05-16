#ifndef GENERATORS_H
#define GENERATORS_H

#include <iostream>
#include <string>
#include <functional>
#include <vector>
#include <iterator>
#include <cmath>

template <typename Iterable>
auto pairwise(const Iterable& iterable);

std::function<int()> int_generator();

std::function<std::string()> string_generator();

#endif
