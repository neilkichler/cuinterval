#pragma once

#include <cuinterval/interval.h>

#include <concepts>

static_assert(std::regular<cu::interval<int>>);
static_assert(std::regular<cu::interval<float>>);
static_assert(std::regular<cu::interval<double>>);
