#ifndef _ANNOTATION_H_
#define _ANNOTATION_H_

#include <vector>
#include <string>
#include <optional>

// Segment, Track, Labelを持つRecord構造体を定義
struct Record {
    double start;
    double end;
    int track;
    std::string label;
};

class Annotation {
public:
    static Annotation fromRecords(
        const std::vector<Record>& records,
        const std::optional<std::string>& uri = std::nullopt,
        const std::optional<std::string>& modality = std::nullopt
    );

    // その他のメンバー関数やプロパティ
};

#endif