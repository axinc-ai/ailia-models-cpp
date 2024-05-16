#ifndef FEATURE_H
#define FEATURE_H

#include <vector>
#include <string>
#include <stdexcept>
#include <variant>
#include <optional>

#include "segment.h"  // Segment クラスのヘッダーファイルを想定

class SlidingWindowFeature {
private:
    SlidingWindow sliding_window;  // この型の定義が必要です。
    std::vector<std::vector<double>> data;
    std::vector<std::string> labels;
    int index;

public:
    SlidingWindowFeature(const std::vector<std::vector<double>>& data, const SlidingWindow& sliding_window, const std::vector<std::string>& labels = {});

    int __len__() const;
    Segment extent() const;
    int dimension() const;
    int getNumber() const;
    int getDimension() const;
    Segment getExtent() const;
    std::vector<double>& operator[](int i);
    SlidingWindowFeature& __iter__();
    std::tuple<Segment, std::vector<double>> __next__();
    std::tuple<Segment, std::vector<double>> next();
    std::vector<std::variant<std::vector<double>, std::tuple<std::vector<double>, Segment>>> iterfeatures(bool window = false);
    std::variant<std::vector<std::vector<double>>, SlidingWindowFeature> crop(const Segment& focus, const std::string& mode = "loose", const std::optional<double>& fixed = std::nullopt, bool return_data = true);
    const std::vector<std::vector<double>>& getData() const;
    SlidingWindowFeature operator+(const SlidingWindowFeature& other) const;
    SlidingWindowFeature operator-(const SlidingWindowFeature& other) const;
    SlidingWindowFeature operator*(const SlidingWindowFeature& other) const;
    SlidingWindowFeature operator/(const SlidingWindowFeature& other) const;
    double linearInterpolate(double x, const std::vector<double>& x_values, const std::vector<double>& y_values) const;
    SlidingWindowFeature align(const SlidingWindowFeature& to) const;
};

#endif // FEATURE_H
