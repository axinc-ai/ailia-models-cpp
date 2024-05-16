#include <iostream>
#include <vector>
#include <algorithm>
#include <sstream>
#include <iomanip>


class Segment {
private:
    double start;
    double end;
    static double precision;
    static bool auto_round;

public:
    // コンストラクタ
    Segment(double start = 0.0, double end = 0.0, double precision = 1e-6, bool auto_round = false) : start(start), end(end) { // __init__
        if (auto_round) {
            this->start = std::round(start / precision) * precision;
            this->end = std::round(end / precision) * precision;
        }
    }

    // アクセッサメソッド（self.startのように用いるために定義）
    double getStart() const {
        return start;
    }

    double getEnd() const {
        return end;
    }

    // 比較演算子のオーバーロード（@dataclass(frozen=True, order=True)のため）
    bool operator==(const Segment& other) const {
        return (std::fabs(start - other.start) < precision) &&
               (std::fabs(end - other.end) < precision);
    }
    bool operator!=(const Segment& other) const {
        return !(*this == other);
    }
    bool operator<(const Segment& other) const {
        if (std::fabs(start - other.start) < precision) {
            return end < other.end;
        }
        return start < other.start;
    }
    // セグメント情報の出力
    friend std::ostream& operator<<(std::ostream& os, const Segment& segment) {
        os << "Segment(" << segment.start << ", " << segment.end << ")";
        return os;
    }

    // 精度設定の静的メソッド
    static void set_precision(int ndigits) {
        if (ndigits == -1) {
            auto_round = false;
            precision = 1e-6; // 1 μs
        } else {
            auto_round = true;
            precision = std::pow(10, -ndigits);
        }
    }

    // セグメントの「空」判定
    bool __bool__() const {
        return (end - start) < precision;
    }

    // セグメントの空かどうかを判定する演算子
    operator bool() const { //Segmentのインスタンスに対して，「!segment」のような使い方を可能にするために追加
        return (end - start) < precision;
    }

    void __post_init__() {
        if (auto_round) {
            this->start = std::round(this->start / precision) * precision;
            this->end = std::round(this->end / precision) * precision;
        }
    }

    double duration() const {
        return __bool__() ? end - start : 0.0;
    }

    double middle() const {
        return __bool__() ? 0.5 * (start + end) : 0.0;
    }

    std::vector<double>::iterator __iter_begin__() { //__iter__
        static std::vector<double> bounds = { start, end };
        return bounds.begin();
    }

    std::vector<double>::iterator __iter_end__() { //__iter__
        static std::vector<double> bounds = { start, end };
        return bounds.end();
    }

    Segment copy() const {
        return *this;
    }

    // ------------------------------------------------------- 
    // Inclusion (in), intersection (&), union (|) and gap (^) 
    // ------------------------------------------------------- 

    // 包含判定
    bool contains(const Segment& other) const {
        return (start <= other.start) && (end >= other.end);
    }

    // 交差判定
    Segment __and__(const Segment& other) const {
        double newStart = std::max(start, other.start);
        double newEnd = std::min(end, other.end);

        if (newStart < newEnd) {
            return Segment(newStart, newEnd);
        } else {
            return Segment(); // 空のセグメントを返す
        }
    }

    // セグメントが他のセグメントと交差するか判定するメソッド
    bool intersects(const Segment& other) const {
        return (start < other.start && other.start < end - precision) ||
            (start > other.start && start < other.end - precision) ||
            (start == other.start);
    }

    // セグメントが特定の時間tと重なるか判定するメソッド
    bool overlaps(double t) const {
        return start <= t && end >= t;
    }

    // ユニオン操作（合併）
    Segment  __or__(const Segment& other) const { // __or__ → operator|
        if (!__bool__()) return other; // 現在のセグメントが空の場合、otherを返す
        if (!other.__bool__()) return *this; // otherが空の場合、現在のセグメントを返す

        double newStart = std::min(start, other.start);
        double newEnd = std::max(end, other.end);
        return Segment(newStart, newEnd); // 新しいセグメントを返す
    }

    // ギャップ操作（差分）
    Segment __xor__(const Segment& other) const { // __xor__ → operator^
        if (!__bool__() || !other.__bool__()) {
            throw std::invalid_argument("The gap between a segment and an empty segment is not defined.");
        }

        double newStart = std::min(end, other.end);
        double newEnd = std::max(start, other.start);
        if (newStart < newEnd) {
            return Segment(newStart, newEnd); // 新しいセグメントを返す
        } else {
            return Segment(); // 空のセグメントを返す
        }
    }

    //秒を時間表記の文字列に変換
    //与えられた秒数を hh:mm:ss.mmm 形式の文字列に変換
    std::string _str_helper(double seconds) const {
        bool negative = seconds < 0;
        if (negative) {
            seconds = -seconds;
        }
        
        int hours = static_cast<int>(seconds / 3600);
        seconds -= hours * 3600;
        int minutes = static_cast<int>(seconds / 60);
        seconds -= minutes * 60;
        double fractional = seconds - static_cast<int>(seconds);
        seconds = static_cast<int>(seconds);

        std::stringstream ss;
        if (negative) {
            ss << "-";
        }
        ss << std::setw(2) << std::setfill('0') << hours << ":"
        << std::setw(2) << std::setfill('0') << minutes << ":"
        << std::setw(2) << std::setfill('0') << seconds << "."
        << std::setw(3) << static_cast<int>(fractional * 1000);

        return ss.str();
    }

    // セグメントが空でない場合にフォーマットされた開始時刻と終了時刻を出力
    std::string __str__() const {
        if (__bool__()) {
            return "[" + _str_helper(start) + " --> " + _str_helper(end) + "]";
        }
        return "[]";
    }

    // プログラムが読み取りやすい表現を出力
    // セグメントの開始時刻と終了時刻を <Segment(開始時刻, 終了時刻)> の形式で出力
    std::string __repr__() const {
        std::stringstream ss;
        ss << "<Segment(" << start << ", " << end << ")>";
        return ss.str();
    }

};



class SlidingWindow {
private:
    double duration;
    double step;
    double start;
    double end;
    int index;

public:
    // コンストラクタ
    SlidingWindow(double duration = 0.030, double step = 0.010, double start = 0.000, double end = std::numeric_limits<double>::infinity()) 
    : duration(duration), step(step), start(start), end(end), index(-1) {
        if (duration <= 0) {
            throw std::invalid_argument("'duration' must be a float > 0.");
        }
        if (step <= 0) {
            throw std::invalid_argument("'step' must be a float > 0.");
        }
        if (end != std::numeric_limits<double>::infinity() && end <= start) {
            throw std::invalid_argument("'end' must be greater than 'start'.");
        }

        this->duration = duration;
        this->step = step;
        this->start = start;
        this->end = end;
    }

    // アクセッサメソッド
    double getStart() const {
        return start;
    }

    double getEnd() const {
        return end;
    }

    double getStep() const {
        return step;
    }

    double getDuration() const {
        return duration;
    }

    // 最も近いフレームのインデックスを計算
    int closest_frame(double t) const {
        return static_cast<int>(std::rint(
            (t - start - 0.5 * duration) / step
        ));
    }

    // 特定の期間からフレーム数を計算
    int samples(double from_duration, const std::string& mode = "strict") const {
        if (mode == "strict") {
            return static_cast<int>(std::floor((from_duration - duration) / step)) + 1;
        } else if (mode == "loose") {
            return static_cast<int>(std::floor((from_duration + duration) / step));
        } else if (mode == "center") {
            return static_cast<int>(std::rint(from_duration / step));
        }
        throw std::invalid_argument("Invalid mode specified.");
    }

    std::vector<int> crop(const Segment& focus, const std::string& mode = "loose", double fixed = 0) const {
        double focusStart = focus.getStart();
        double focusEnd = focus.getEnd();
        int i, j;

        if (mode == "loose") {
            i = static_cast<int>(std::ceil((focusStart - duration - start) / step));
            j = static_cast<int>(std::floor((focusEnd - start) / step));
        } else if (mode == "strict") {
            i = static_cast<int>(std::ceil((focusStart - start) / step));
            j = static_cast<int>(std::floor((focusEnd - duration - start) / step));
        } else if (mode == "center") {
            i = closest_frame(focusStart);
            j = closest_frame(focusEnd);
        } else {
            throw std::invalid_argument("'mode' must be one of {'loose', 'strict', 'center'}.");
        }

        if (fixed > 0) {
            int n = samples(fixed, mode);
            return std::vector<int>(i, i + n);
        } else {
            return std::vector<int>(i, j + 1);
        }
    }

    
    // セグメントをフレーム範囲に変換
    std::pair<int, int> segment_to_range(const Segment& segment) const {
        int i0 = closest_frame(segment.getStart());
        int n = static_cast<int>(std::ceil(segment.duration() / this->step)) + 1;
        return {i0, n};
    }

    // セグメントをフレーム範囲に変換
    std::pair<int, int> segmentToRange(const Segment& segment) const {
        int i0 = closest_frame(segment.getStart());
        int n = static_cast<int>(std::ceil(segment.duration() / this->step)) + 1;
        return {i0, n};
    }

    // フレーム範囲をセグメントに変換
    Segment range_to_segment(int i0, int n) const {
        double start = this->start + (i0 - 0.5) * this->step + 0.5 * this->duration;
        double end = start + n * this->step;
        if (i0 == 0) {
            start = this->start;
        }
        return Segment(start, end);
    }

    // フレーム範囲をセグメントに変換
    Segment rangeToSegment(int i0, int n) const {
        double start = this->start + (i0 - 0.5) * this->step + 0.5 * this->duration;
        double end = start + n * this->step;
        if (i0 == 0) {
            start = this->start;
        }
        return Segment(start, end);
    }

    // サンプル数から期間を計算する
    double samples_to_duration(int n_samples) const {
        Segment tempSegment = range_to_segment(0, n_samples);
        return tempSegment.duration();
    }

    // サンプル数から期間を計算する
    double samplesToDuration(int n_samples) const {
        Segment tempSegment = range_to_segment(0, n_samples);
        return tempSegment.duration();
    }

    // 期間からサンプル数を計算する
    int duration_to_samples(double duration) const {
        Segment tempSegment(0, duration);
        return segment_to_range(tempSegment).second;
    }

    // 期間からサンプル数を計算する
    int durationToSamples(double duration) const {
        Segment tempSegment(0, duration);
        return segment_to_range(tempSegment).second;
    }

    // 指定されたインデックスにあるスライディングウィンドウを取得
    // Segment __getitem__(int i) const {
    //     double segmentStart = start + i * step;
    //     if (segmentStart >= end) {
    //         // 範囲外の場合、空のセグメントを返す
    //         return Segment();
    //     }
    //     return Segment(segmentStart, segmentStart + duration);
    // }

    Segment& operator[](int i) { //__getitem__
        double segmentStart = start + i * step;
        if (segmentStart >= end || i < 0) {
            // 範囲外の場合、例外を投げる
            throw std::out_of_range("Index out of range");
        }
        return *new Segment(segmentStart, segmentStart + duration);  // 注意: この方法はメモリリークを引き起こす可能性があります。
    }

    // 次のスライディングウィンドウを取得するメソッド
    Segment next() {
        ++index;
        Segment window = (*this)[index];
        if (!window) { // Segmentの__bool__メソッドが false を返すとき
            throw std::out_of_range("No more segments available.");
        }
        return window;
    }

    // イテレーション開始
    SlidingWindow& __iter_begin__() { //__iter__
        index = 0;
        return *this;
    }

    // イテレーション終了
    SlidingWindow& __iter_end__() { //__iter__
        index = static_cast<int>((end - start) / step);
        return *this;
    }

    // ウィンドウの数を計算
    int __len__() const {
        if (std::isinf(end)) {
            throw std::runtime_error("infinite sliding window.");
        }
        return static_cast<int>((end - start) / step) + 1;
    }

    // コピー関数
    SlidingWindow copy() const {
        return SlidingWindow(duration, step, start, end);
    }

    std::vector<Segment> operator()(const Segment& support, bool align_last = false) { //__call__
        std::vector<Segment> result;

        double current_start = start;
        while (current_start + duration <= support.getEnd()) {
            double current_end = current_start + duration;

            if (current_end > support.getEnd() && !align_last) {
                break;
            }

            result.push_back(Segment(current_start, current_end));
            current_start += step;
        }

        if (align_last && (support.getEnd() - duration >= support.getStart())) {
            double last_start = support.getEnd() - duration;
            if (result.empty() || result.back().getEnd() != support.getEnd()) {
                result.push_back(Segment(last_start, support.getEnd()));
            }
        }

        return result;
    }

    
};