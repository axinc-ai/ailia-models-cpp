#include <iostream>
#include <string>
#include <functional>
#include <vector>
#include <iterator>


template <typename Iterable>
auto pairwise(const Iterable& iterable) {
    std::vector<std::pair<typename Iterable::const_iterator, typename Iterable::const_iterator>> result;

    auto first = iterable.begin();
    auto second = first;
    if (first != iterable.end()) {
        ++second;
    }

    while (second != iterable.end()) {
        result.emplace_back(first, second);
        ++first;
        ++second;
    }

    return result;
}


std::function<int()> int_generator() {
    static int value = 0; // 静的ローカル変数による状態保持
    return []() -> int {
        return value++;
    };
}


std::function<std::string()> string_generator() {
    static int length = 1; // 最初は1文字
    static long long current = 0; // 現在の数値カウンター

    return []() -> std::string {
        static const int base = 26;
        static const char baseChar = 'A';

        long long temp = current++;
        std::string result;

        while (temp >= 0) {
            result = char(baseChar + (temp % base)) + result;
            temp = temp / base - 1;
        }

        if (current == pow(base, length)) {
            length++;
            current = 0;
        }

        return result;
    };
}
