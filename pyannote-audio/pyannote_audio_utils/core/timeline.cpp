#include <vector>
#include <set>
#include <algorithm>
#include <string>
#include <sstream>
#include <fstream>
#include <iomanip>

#include "types.h"
#include "segment.h"
#include "annotation.h"

class Timeline {
private:
    std::vector<Segment> segmentsVector; // segments_list_　セグメントのベクター
    std::set<Segment> segmentsSet;       // segments_set_　重複を避け、順序を保つためのセット
    std::string uri;                     // リソースのURI

public:
    
    Timeline(const std::string& uri) : uri(uri) {} // URIのみを引数に取るコンストラクタ

    // コンストラクタ
    Timeline(const std::vector<Segment>& segments = {}, const std::string& uri = "") //__init__
        : uri(uri) {
        for (const auto& segment : segments) {
            add(segment);
        }
    }

    // URIを設定するメンバ関数
    void setUri(const std::string& newUri) { //__init__
        uri = newUri;
    }

    // セグメントを取得するメンバ関数
    const std::vector<Segment>& getSegments() const { //__init__
        return segmentsVector;
    }


    // タイムラインのセグメント数を取得
    size_t __len__() const {
        return segmentsSet.size();
    }

    // タイムラインが空かどうかを判定
    operator bool() const { //__nonzero__，__bool__
        return !segmentsSet.empty();
    }

    // イテレータを提供 (セグメントの開始によるソート順)
    std::vector<Segment>::const_iterator begin() const { //__iter__
        return segmentsVector.begin();
    }

    std::vector<Segment>::const_iterator end() const { //__iter__
        return segmentsVector.end();
    }

    // インデックスによるセグメントのアクセス
    const Segment& operator[](size_t index) const { //__getitem__
        return segmentsVector.at(index);
    }

    // タイムラインの比較 (等価性)
    bool operator==(const Timeline& other) const { //__eq__
        return segmentsSet == other.segmentsSet;
    }

    bool operator!=(const Timeline& other) const { //__ne__
        return segmentsSet != other.segmentsSet;
    }

    // セグメントのインデックスを返す
    int index(const Segment& segment) const {
        auto it = std::find(segmentsVector.begin(), segmentsVector.end(), segment);
        if (it == segmentsVector.end()) {
            throw std::runtime_error("Segment not found in timeline.");
        }
        return std::distance(segmentsVector.begin(), it);
    }

    // セグメントを追加するメンバ関数
    void add(const Segment& segment) {
        if (segment && segmentsSet.find(segment) == segmentsSet.end()) {
            segmentsSet.insert(segment);
            segmentsVector.push_back(segment);
            std::sort(segmentsVector.begin(), segmentsVector.end());
        }
    }

    // セグメントを削除する
    void remove(const Segment& segment) {
        auto it = segmentsSet.find(segment);
        if (it != segmentsSet.end()) {
            segmentsSet.erase(it);
            segmentsVector.erase(std::remove(segmentsVector.begin(), segmentsVector.end(), segment), segmentsVector.end());
        }
    }

    void discard(const Segment& segment) {
        auto it = segmentsSet.find(segment);
        if (it != segmentsSet.end()) {
            segmentsSet.erase(it);
            segmentsVector.erase(std::remove(segmentsVector.begin(), segmentsVector.end(), segment), segmentsVector.end());
        }
    }

    // 他のタイムラインのセグメントを組み込む
    Timeline& operator|=(const Timeline& other) { //__ior__
        for (const auto& segment : other.segmentsVector) {
            add(segment);
        }
        return *this;
    }

    // 他のタイムラインのセグメントを組み込む
    Timeline& update(const Timeline& other) {
        for (const auto& segment : other.segmentsVector) {
            add(segment);
        }
        return *this;
    }

    // 他のタイムラインとの結合を新しいタイムラインで返す
    Timeline operator|(const Timeline& other) const { //__or__
        Timeline result = *this;
        result |= other;
        return result;
    }

    // 他のタイムラインとの結合を新しいタイムラインで返す
    Timeline unionWith(const Timeline& other) const {
        Timeline result = *this;
        result |= other;
        return result;
    }

    std::vector<std::pair<Segment, Segment>> co_iter(const Timeline& other) const {
        std::vector<std::pair<Segment, Segment>> result;
        for (const auto& segment1 : segmentsVector) {
            for (const auto& segment2 : other.segmentsVector) {
                if (segment1.intersects(segment2)) {
                    result.emplace_back(segment1, segment2);
                }
            }
        }
        return result;
    }

    std::vector<Segment> crop_iter(const Support& support, const std::string& mode = "intersection") const {
        std::vector<Segment> cropped;
        
        std::visit([&](auto&& arg) {
            using T = std::decay_t<decltype(arg)>;
            if constexpr (std::is_same_v<T, Timeline*>) {
                const auto& supportSegments = arg->getSegments();
                if (mode == "loose") {
                    for (const auto& segment : segmentsVector) {
                        for (const auto& other : supportSegments) {
                            if (segment.intersects(other)) {
                                cropped.push_back(segment);
                                break; // Break after the first intersection for loose mode
                            }
                        }
                    }
                } else if (mode == "strict") {
                    for (const auto& segment : segmentsVector) {
                        bool allContained = true;
                        for (const auto& other : supportSegments) {
                            if (!other.contains(segment)) {
                                allContained = false;
                                break;
                            }
                        }
                        if (allContained) cropped.push_back(segment);
                    }
                } else if (mode == "intersection") {
                    for (const auto& segment : segmentsVector) {
                        for (const auto& other : supportSegments) {
                            if (segment.intersects(other)) {
                                cropped.push_back(segment & other);
                            }
                        }
                    }
                }
            } else if constexpr (std::is_same_v<T, Segment*>) {
                // Segment pointer がサポートとして与えられた場合の処理
                const Segment& singleSupport = *arg;
                if (mode == "loose") {
                    for (const auto& segment : segmentsVector) {
                        if (segment.intersects(singleSupport)) {
                            cropped.push_back(segment);
                        }
                    }
                } else if (mode == "strict") {
                    for (const auto& segment : segmentsVector) {
                        if (singleSupport.contains(segment)) {
                            cropped.push_back(segment);
                        }
                    }
                } else if (mode == "intersection") {
                    for (const auto& segment : segmentsVector) {
                        if (segment.intersects(singleSupport)) {
                            cropped.push_back(segment & singleSupport);
                        }
                    }
                }
            }
        }, *support);
        return cropped;
    }

    Timeline crop(const Support& support, const std::string& mode = "intersection") const {
        std::vector<Segment> croppedSegments = crop_iter(support, mode);
        return Timeline(croppedSegments, uri);
    }

    std::vector<Segment> overlapping(double t) const { //overlapping_iterも兼ねる（C++ではイテレータを定義できず，元々ベクターを返しているため）
        std::vector<Segment> result;
        for (const auto& segment : segmentsVector) {
            if (segment.overlaps(t)) {
                result.push_back(segment);
            }
        }
        return result;
    }


    Timeline get_overlap() const {
        Timeline overlaps_tl(uri); // 新しいタイムラインを作成
        for (const auto& s1 : segmentsVector) {
            for (const auto& s2 : segmentsVector) {
                if (&s1 != &s2 && s1.intersects(s2)) {
                    overlaps_tl.add(s1 & s2); // 重なるセグメントの交差部分を追加
                }
            }
        }
        return overlaps_tl; // オーバーラップするタイムラインを返す
    }

    Timeline extrude(const Timeline& removed, const std::string& mode = "intersection") const {
        Timeline result(uri); // 結果として返す新しいタイムラインを生成

        // サポートするタイムラインの取得（除去する部分）
        Timeline support = removed.support(); // `support()` は、removedに含まれる全セグメントの集合（重なり合わない形で）を返す関数と仮定

        // タイムライン内の各セグメントに対して処理を行う
        for (const Segment& segment : segmentsVector) {
            bool isRemoved = false;  // セグメントが削除されるかどうかを判定するフラグ

            // サポートするタイムラインとの交差点を確認
            for (const Segment& removeSegment : support.getSegments()) {
                if (mode == "strict" && removeSegment.contains(segment)) {
                    isRemoved = true;  // strictモードでは完全に含まれる場合のみ削除
                    break;
                } else if (mode == "loose" && segment.intersects(removeSegment)) {
                    isRemoved = true;  // looseモードでは交差する場合に削除
                    break;
                } else if (mode == "intersection") {
                    if (segment.intersects(removeSegment)) {
                        Segment intersection = segment & removeSegment;
                        Segment firstPart(segment.getStart(), intersection.getStart());
                        Segment secondPart(intersection.getEnd(), segment.getEnd());
                        if (firstPart) result.add(firstPart);
                        if (secondPart) result.add(secondPart);
                        isRemoved = true;
                        break;
                    }
                }
            }

            if (!isRemoved) {
                result.add(segment);  // 除去リストに含まれない場合、結果タイムラインに追加
            }
        }

        return result;
    }

    std::string __str__() const {
        std::stringstream ss;
        ss << "[";
        for (auto it = segmentsVector.begin(); it != segmentsVector.end(); ++it) {
            if (it != segmentsVector.begin()) {
                ss << " ";
            }
            ss << it->__str__();
        }
        ss << "]";
        return ss.str();
    }

    std::string __repr__() const {
        std::stringstream ss;
        ss << "<Timeline(uri=" << uri << ", segments=" << __str__() << ")>";
        return ss.str();
    }

    bool __contains__(const Segment& segment) const {
        return segmentsSet.find(segment) != segmentsSet.end();
    }

    // 空のコピーを返す
    Timeline empty() const {
        return Timeline({}, uri); // 新しいTimelineオブジェクトを作成し、現在のURIを使用する
    }

    // このタイムラインが他のタイムラインを完全に覆っているかどうかを確認
    bool covers(const Timeline& other) const {
        // otherの各セグメントが、このタイムラインによって覆われているかどうかをチェック
        for (const auto& otherSegment : other.segmentsVector) {
            bool covered = false;
            for (const auto& segment : segmentsVector) {
                if (segment.contains(otherSegment)) {
                    covered = true;
                    break;
                }
            }
            if (!covered) return false; // 少なくとも1つのセグメントが覆われていない場合、falseを返す
        }
        return true; // すべてのセグメントが覆われている場合、trueを返す
    }

    // タイムラインのコピーを作成し、必要に応じてセグメントに関数を適用
    Timeline copy(std::function<Segment(Segment)> segmentFunc = nullptr) const {
        std::vector<Segment> newSegments;
        for (const Segment& segment : segmentsVector) {
            // 関数が提供されている場合は、その関数をセグメントに適用
            newSegments.push_back(segmentFunc ? segmentFunc(segment) : segment);
        }
        return Timeline(newSegments, uri); // 新しいセグメントと同じURIで新しいTimelineを作成
    }

    // タイムラインの範囲（最小の包括セグメント）を求める
    Segment extent() const {
        if (segmentsVector.empty()) {
            return Segment();
        }
        double start = std::numeric_limits<double>::max();
        double end = std::numeric_limits<double>::min();
        for (const auto& segment : segmentsVector) {
            if (segment.getStart() < start) start = segment.getStart();
            if (segment.getEnd() > end) end = segment.getEnd();
        }
        return Segment(start, end);
    }

    // サポートセグメントを生成する
    // タイムライン内で連続するセグメントを結合する関数
    std::vector<Segment> support_iter(double collar = 0.0) const {
        std::vector<Segment> result;
        if (segmentsVector.empty()) {
            return result; // 空のタイムラインの場合は空の結果を返す
        }

        // 最初のセグメントを初期値として設定
        Segment current = segmentsVector[0];

        for (size_t i = 1; i < segmentsVector.size(); ++i) {
            const auto& next_segment = segmentsVector[i];

            // 次のセグメントとの間のギャップを計算
            double gap = next_segment.getStart() - current.getEnd();

            if (gap <= collar) {
                // ギャップが許容範囲内であればセグメントを結合
                current = Segment(current.getStart(), next_segment.getEnd());
            } else {
                // ギャップが許容範囲を超えていた場合は、現在のセグメントを結果に追加し、新しいセグメントを開始
                result.push_back(current);
                current = next_segment;
            }
        }

        // 最後のセグメントを追加
        result.push_back(current);

        return result;
    }


    // タイムラインのサポートを取得する
    Timeline support(double collar = 0.0) const {
        std::vector<Segment> supportSegments = support_iter(collar);
        return Timeline(supportSegments, uri);
    }

    //各セグメントの長さを計算し合計
    double duration() const {
        double total_duration = 0.0;
        for (const auto& segment : support()) {  // support() は既に定義済みと仮定
            total_duration += segment.duration();
        }
        return total_duration;
    }


    std::vector<Segment> gaps_iter(const Support& support) const {
        std::vector<Segment> result;

        // if (!support) {
        //     return result;
        // }

        std::visit([&](auto&& arg) {
            using T = std::decay_t<decltype(arg)>;
            if constexpr (std::is_same_v<T, Segment*>) {
                // Segment* の処理
                Segment* supportSegment = arg;
                double end = supportSegment->getStart();
                auto cropped = crop(Support(supportSegment), "intersection").support(0.0);
                for (const auto& segment : cropped.getSegments()) {
                    Segment gap(end, segment.getStart());
                    if (gap) {
                        result.push_back(gap);
                    }
                    end = segment.getEnd();
                }
                // 最後のギャップを追加
                Segment finalGap(end, supportSegment->getEnd());
                if (finalGap) {
                    result.push_back(finalGap);
                }
            } else if constexpr (std::is_same_v<T, Timeline*>) {
                // Timeline* の処理
                Timeline* supportTimeline = arg;
                for (const auto& supportSegment : supportTimeline->getSegments()) {
                    auto gapsForSegment = gaps_iter(Support(&supportSegment));
                    result.insert(result.end(), gapsForSegment.begin(), gapsForSegment.end());
                }
            }
        }, *support); // std::optional から std::variant を取り出す

        return result;
    }

    
    // 関数定義
    Timeline gaps(const Support& support = Support{}) const {
        Support usedSupport = support ? support : Support(new Segment(0, std::numeric_limits<double>::max()));
        std::vector<Segment> gapsSegments = gaps_iter(usedSupport);
        return Timeline(gapsSegments, uri);
    }
    

    // タイムラインから重なりのないセグメントの新しいタイムラインを生成
    Timeline segmentation() const {
        Timeline result(uri);  // 新しいタイムラインを作成
        auto support = this->support();  // 重なりがないサポートタイムラインを取得

        std::set<double> timestamps;  // セグメントの境界を保持するセット
        for (const auto& segment : segmentsVector) {
            timestamps.insert(segment.getStart());
            timestamps.insert(segment.getEnd());
        }

        if (timestamps.empty()) {
            return result;  // 境界がない場合、空のタイムラインを返す
        }

        double prev = *timestamps.begin();
        for (auto it = std::next(timestamps.begin()); it != timestamps.end(); ++it) {
            Segment seg(prev, *it);
            // overlapping() が空でない場合に true と評価されるように修正
            if (!support.overlapping(seg.middle()).empty()) {
                result.add(seg);
            }
            prev = *it;
        }

        return result;
    }



    //ここは，annotation.cppの定義が終わったのち追加
    /*
    Annotation toAnnotation(const std::string& generatorType, const std::string& modality) const {
        Annotation annotation(uri, modality);
        LabelGenerator generator(generatorType);  // ジェネレータを初期化

        for (const auto& segment : segmentsVector) {
            annotation.addLabel(segment, generator.nextLabel());
        }

        return annotation;
    }
    */


    // UEMフォーマットでのシリアライズを行う関数
    std::string to_uem() const { //_iter_uemも兼ねる
        std::ostringstream oss;
        std::string uri_str = uri.empty() ? "<NA>" : uri;
        if (uri_str.find(' ') != std::string::npos) {
            throw std::runtime_error("Space-separated UEM file format does not allow file URIs containing spaces.");
        }
        for (const auto& segment : segmentsVector) {
            oss << uri_str << " 1 " << std::fixed << std::setprecision(3) 
                << segment.getStart() << " " << segment.getEnd() << "\n";
        }
        return oss.str();
    }

    // UEMフォーマットでファイルに書き込む関数
    void write_uem(const std::string& filename) const {
        std::ofstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open file for writing.");
        }
        file << to_uem();
        file.close();
    }


    // _repr_png_ は colab用なのでなし




};
