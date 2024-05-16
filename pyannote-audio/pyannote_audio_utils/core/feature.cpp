#include <vector>
#include <string>
#include <stdexcept>
#include <variant>
#include <optional>

#include "segment.h"

class SlidingWindowFeature {
private:
    SlidingWindow sliding_window;
    std::vector<std::vector<double>> data;
    std::vector<std::string> labels;
    int index;

public:
    SlidingWindowFeature(
        const std::vector<std::vector<double>>& data,
        const SlidingWindow& sliding_window,
        const std::vector<std::string>& labels = {}
    ) : data(data), sliding_window(sliding_window), labels(labels), index(-1) {}

    int __len__() const {
        return data.size();
    }

    Segment extent() const {
        return sliding_window.range_to_segment(0, __len__());
    }

    int dimension() const {
        if (!data.empty()) {
            return data[0].size();
        }
        return 0;
    }

    // 新しいメソッドを追加
    int getNumber() const {
        std::cerr << "This is deprecated in favor of `__len__`" << std::endl;
        return __len__();
    }

    int getDimension() const {
        std::cerr << "This is deprecated in favor of `dimension` property" << std::endl;
        return dimension();
    }

    Segment getExtent() const {
        std::cerr << "This is deprecated in favor of `extent` property" << std::endl;
        return extent();
    }

    std::vector<double>& operator[](int i) { //__getitem__
        if (i < 0 || i >= static_cast<int>(data.size())) {
            throw std::out_of_range("Index out of range");
        }
        return data[i];
    }

    SlidingWindowFeature&  __iter__() {
        index = -1;
        return *this;
    }

    std::tuple<Segment, std::vector<double>> __next__() {
        ++index;
        if (index >= __len__()) {
            throw std::out_of_range("StopIteration");
        }
        return std::make_tuple(sliding_window.range_to_segment(index, 1), data[index]);
    }

    std::tuple<Segment, std::vector<double>> next() {
        ++index;
        if (index >= __len__()) {
            throw std::out_of_range("StopIteration");
        }
        return std::make_tuple(sliding_window.range_to_segment(index, 1), data[index]);
    }

    std::vector<std::variant<std::vector<double>, std::tuple<std::vector<double>, Segment>>> iterfeatures(bool window = false) {
        std::vector<std::variant<std::vector<double>, std::tuple<std::vector<double>, Segment>>> result;
        for (int i = 0; i < __len__(); ++i) {
            if (window) {
                result.emplace_back(std::make_tuple(data[i], sliding_window.range_to_segment(i, 1)));
            } else {
                result.emplace_back(data[i]);
            }
        }
        return result;
    }

    std::variant<std::vector<std::vector<double>>, SlidingWindowFeature> crop(
    const Segment& focus,
    const std::string& mode = "loose",
    const std::optional<double>& fixed = std::nullopt,
    bool return_data = true
    ) {
        if (!return_data && fixed.has_value()) {
            throw std::invalid_argument("\"fixed\" cannot be set when \"return_data\" is set to False.");
        }

        std::vector<int> indices = sliding_window.crop(focus, mode, fixed.value_or(0.0));
        std::vector<std::vector<double>> cropped_data;

        for (int idx : indices) {
            if (idx < 0 || idx >= __len__()) {
                continue; // Skip indices that are out of bounds
            }
            cropped_data.push_back(data[idx]);
        }

        if (return_data) {
            return cropped_data;
        } else {
            // Create a new SlidingWindow based on the first and last indices in the range
            double new_start = sliding_window[indices.front()].getStart();
            double new_end = sliding_window[indices.back()].getEnd();
            SlidingWindow new_sliding_window(new_start, sliding_window.getStep(), new_start, new_end);
            return SlidingWindowFeature(cropped_data, new_sliding_window, labels);
        }
    }

    //_repr_png_はColab用のためなし

    const std::vector<std::vector<double>>& getData() const { //__array__
        return data;
    }

    // 演算子オーバーロード例
    SlidingWindowFeature operator+(const SlidingWindowFeature& other) const { //__array_ufunc__
        std::vector<std::vector<double>> result_data(data.size(), std::vector<double>(data[0].size()));
        for (size_t i = 0; i < data.size(); ++i) {
            std::transform(data[i].begin(), data[i].end(), other.data[i].begin(), result_data[i].begin(), std::plus<double>());
        }
        return {result_data, sliding_window, labels};
    }

    // 減算オーバーロード
    SlidingWindowFeature operator-(const SlidingWindowFeature& other) const { //__array_ufunc__
        std::vector<std::vector<double>> result_data(data.size(), std::vector<double>(data[0].size()));
        for (size_t i = 0; i < data.size(); ++i) {
            std::transform(data[i].begin(), data[i].end(), other.data[i].begin(), result_data[i].begin(), std::minus<double>());
        }
        return {result_data, sliding_window, labels};
    }

    // 乗算オーバーロード
    SlidingWindowFeature operator*(const SlidingWindowFeature& other) const { //__array_ufunc__
        std::vector<std::vector<double>> result_data(data.size(), std::vector<double>(data[0].size()));
        for (size_t i = 0; i < data.size(); ++i) {
            std::transform(data[i].begin(), data[i].end(), other.data[i].begin(), result_data[i].begin(), std::multiplies<double>());
        }
        return {result_data, sliding_window, labels};
    }

    // 除算オーバーロード
    SlidingWindowFeature operator/(const SlidingWindowFeature& other) const { //__array_ufunc__
        std::vector<std::vector<double>> result_data(data.size(), std::vector<double>(data[0].size()));
        for (size_t i = 0; i < data.size(); ++i) {
            std::transform(data[i].begin(), data[i].end(), other.data[i].begin(), result_data[i].begin(), [](double x, double y) {
                return y != 0 ? x / y : 0; // ゼロ除算を避ける
            });
        }
        return {result_data, sliding_window, labels};
    }

    double linearInterpolate(double x, const std::vector<double>& x_values, const std::vector<double>& y_values) const { //np.interp
        if (x_values.empty() || y_values.empty() || x_values.size() != y_values.size())
            return std::nan("");  // Return NaN if input is invalid

        auto lower = std::lower_bound(x_values.begin(), x_values.end(), x);
        if (lower == x_values.begin()) {
            return y_values.front();
        } else if (lower == x_values.end()) {
            return y_values.back();
        } else {
            auto i = lower - x_values.begin();
            double x1 = x_values[i - 1], x2 = x_values[i];
            double y1 = y_values[i - 1], y2 = y_values[i];
            return y1 + (x - x1) / (x2 - x1) * (y2 - y1);  // Linear interpolation formula
        }
    }

    SlidingWindowFeature align(const SlidingWindowFeature& to) const {
        std::vector<double> old_t, new_t;
        double old_start = sliding_window.getStart();
        double new_start = to.sliding_window.getStart();
        double old_step = sliding_window.getStep();
        double new_step = to.sliding_window.getStep();
        double old_duration = sliding_window.getDuration();
        double new_duration = to.sliding_window.getDuration();
        int old_samples = __len__();
        int new_samples = to.__len__();

        for (int i = 0; i < old_samples; ++i) {
            old_t.push_back(old_start + 0.5 * old_duration + i * old_step);
        }
        for (int i = 0; i < new_samples; ++i) {
            new_t.push_back(new_start + 0.5 * new_duration + i * new_step);
        }

        std::vector<std::vector<double>> new_data(new_samples, std::vector<double>(dimension()));

        for (size_t j = 0; j < dimension(); ++j) {
            std::vector<double> old_data_column;
            for (const auto& row : data) {
                old_data_column.push_back(row[j]);
            }
            for (size_t i = 0; i < new_samples; ++i) {
                new_data[i][j] = linearInterpolate(new_t[i], old_t, old_data_column);
            }
        }

        return {new_data, to.sliding_window, labels};
    }

};