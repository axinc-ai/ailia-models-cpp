#ifndef _TYPES_H_
#define _TYPES_H_

#include <string>
#include <variant>
#include <tuple>
#include <vector>
#include <memory>
#include <type_traits>
#include <optional>

// クラスの前方宣言
class Segment;
class Timeline;
class SlidingWindowFeature;
class Annotation;


// Support と Resource の型定義
using Support = std::optional<std::variant<const Segment*, const Timeline*>>;
using Resource = std::variant<Segment*, Timeline*, SlidingWindowFeature*, Annotation*>;

// TrackName と Key の定義
using TrackName = std::variant<std::string, int>;
// using Key = std::variant<Segment*, std::tuple<Segment*, TrackName>>;

// LabelGeneratorMode と LabelGenerator の定義
enum class LabelGeneratorMode { Int, String };
using Label = std::string;  // Hashable は C++ で std::string または自作のハッシュ可能なクラスで置き換え可能
using LabelGenerator = std::variant<LabelGeneratorMode, std::vector<Label>>;

// CropMode と Alignment の enum 定義
enum class CropMode { Intersection, Loose, Strict };
enum class Alignment { Center, Loose, Strict };

// LabelStyle の定義
using LabelStyle = std::tuple<std::string, int, std::tuple<float, float, float>>;





#endif