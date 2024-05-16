#include <vector>
#include <string>
#include <optional>
#include <map>
#include <fstream>
#include <sstream>
#include <iostream>
#include <memory>
#include <cmath>
#include <limits>


#include "types.h"
#include "segment.h"
#include "timeline.h"
#include "generators.h"
#include "feature.h"

//pythonのDataFrameの代わり
struct Record {
    double start;
    double end;
    int track;
    std::string label;
};


class Annotation
{
private:
    std::optional<std::string> _uri;
    std::optional<std::string> modality;
    std::map<Segment, std::map<TrackName, Label>> _tracks; // Sorted dictionary
    std::map<Label, std::shared_ptr<Timeline>> _labels;
    std::map<Label, bool> _labelNeedsUpdate;
    std::shared_ptr<Timeline> _timeline;
    bool _timelineNeedsUpdate;


public:
    // Annotation fromRecords( //from_dfと対応
    // const std::vector<Record>& records,
    // const std::optional<std::string>& uri,
    // const std::optional<std::string>& modality
    // ) {
    //     // RecordsをもとにAnnotationオブジェクトを構築するロジックをここに実装
    //     // この例では、単純にレコードリストを受け取り、Annotationオブジェクトを返すとしています。
    //     Annotation annotation;
    //     // Annotationの初期化や設定を行う
    //     return annotation;
    // }

    //コンストラクタ
    Annotation(const std::optional<std::string>& uri, const std::optional<std::string>& modality)
        : _uri(uri), modality(modality), _timelineNeedsUpdate(true) {
        _timeline = std::make_shared<Timeline>();
    };

    // URIゲッター
    std::optional<std::string> getUri() const { //uri（@property）
        return _uri;
    }

    // URIセッター
    // label_timelineとget_timelineを用いたコードに要修正
    void setUri(const std::string& uri) { //uri（@uri.setter）
        _uri = uri;
        _timeline->setUri(uri);
        for (auto& label_timeline : _labels) {
            label_timeline.second->setUri(uri);
        }
    }



    void updateLabels() {
        // Identify labels that need to be updated
        std::set<std::string> update;
        for (const auto& label_update : _labelNeedsUpdate) {
            if (label_update.second) {
                update.insert(label_update.first);
            }
        }

        // Accumulate segments for updated labels
        std::map<std::string, std::vector<Segment>> segments;
        for (const auto& track : _tracks) {
            for (const auto& label : track.second) {
                if (update.find(label.second) != update.end()) {
                    segments[label.second].push_back(track.first);
                }
            }
        }

        // Create timelines with accumulated segments for updated labels
        // itertracksを使ったコードに要修正
        for (const auto& label : update) {
            if (!segments[label].empty()) {
                // Use value_or to handle std::optional<std::string>
                std::string uri = _uri.value_or("");
                _labels[label] = std::make_shared<Timeline>(segments[label], uri);
                _labelNeedsUpdate[label] = false;
            } else {
                _labels.erase(label);
                _labelNeedsUpdate.erase(label);
            }
        }
    }
    

    // セグメント数を取得するメソッド
    size_t __len__() const {
        return _tracks.size();
    }

    size_t __nonzero__() const {
        return _tracks.size();
    }

    // Annotationオブジェクトが空かどうかを判断するメソッド
    bool __bool__() const {
        return _tracks.empty();
    }


    // セグメントをイテレートするメソッド
    std::vector<Segment> itersegments() const {
        std::vector<Segment> segments;
        for (const auto& track : _tracks) {
            segments.push_back(track.first);
        }
        std::sort(segments.begin(), segments.end());
        return segments;
    }


    std::vector<std::tuple<Segment, TrackName, std::optional<Label>>> iterTracks(bool yield_label = false) const {
        std::vector<std::tuple<Segment, TrackName, std::optional<Label>>> tracks;
        for (const auto& track_item : _tracks) {
            const Segment& segment = track_item.first;
            for (const auto& track : track_item.second) {
                const TrackName& trackName = track.first;
                if (yield_label) {
                    tracks.emplace_back(segment, trackName, track.second);
                } else {
                    tracks.emplace_back(segment, trackName, std::nullopt);
                }
            }
        }
        return tracks;
    }


    void _updateTimeline() {
        if (_timelineNeedsUpdate) {
            _timeline = std::make_shared<Timeline>(); // Assuming we refresh the entire timeline
            for (const auto& entry : _tracks) {
                for (const auto& sub_entry : entry.second) {
                    _timeline->add(entry.first); // Assuming there is a method to add segments
                }
            }
            _timelineNeedsUpdate = false;
        }
    }

    std::shared_ptr<Timeline> get_timeline(bool makeCopy = true) {
        if (_timelineNeedsUpdate) {
            _updateTimeline();
        }
        if (makeCopy) {
            return std::make_shared<Timeline>(*_timeline); // Deep copy
        }
        return _timeline; // Return the actual internal timeline (Shallow copy)
    }


    // Equality check
    bool operator==(const Annotation& other) const { //__eq__
        if (_tracks.size() != other._tracks.size()) return false;
        auto it1 = _tracks.begin();
        auto it2 = other._tracks.begin();
        while (it1 != _tracks.end()) {
            if (*it1 != *it2) return false;
            ++it1; ++it2;
        }
        return true;
    }

    // Inequality check
    bool operator!=(const Annotation& other) const { //__ne__
        return !(*this == other);
    }

    // Check for inclusion of a segment or a timeline
    bool __contains__(const Segment& segment) {
        // Assuming getTimeline provides a Timeline object that can be queried for inclusion
        auto timeline = get_timeline(false); // Fetching the timeline without making a copy
        return timeline->__contains__(segment);
    }

    bool __contains__(const Timeline& included) {
        auto timeline = get_timeline(false);
        return timeline->__contains__(included);
    }

    // Generates RTTM lines
    std::vector<std::string> _iter_rttm() const {
        std::vector<std::string> lines;
        std::string uri = _uri.value_or("<NA>");

        if (uri.find(' ') != std::string::npos) {
            throw std::runtime_error("URI cannot contain spaces for RTTM format.");
        }

        for (const auto& track_item : _tracks) {
            const Segment& segment = track_item.first;
            for (const auto& track : track_item.second) {
                const std::string& label = track.second;
                if (label.find(' ') != std::string::npos) {
                    throw std::runtime_error("Label cannot contain spaces for RTTM format.");
                }
                std::ostringstream line;
                line << "SPEAKER " << uri << " 1 "
                     << segment.getStart() << " " << segment.duration()
                     << " <NA> <NA> " << label << " <NA> <NA>\n";
                lines.push_back(line.str());
            }
        }
        return lines;
    }

    // Converts annotation to a RTTM format string
    std::string to_rttm() const {
        std::vector<std::string> lines = _iter_rttm();
        std::ostringstream oss;
        for (const auto& line : lines) {
            oss << line;
        }
        return oss.str();
    }

    // Writes RTTM data to a file
    void write_rttm(const std::string& filename) const {
        std::ofstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open file for writing.");
        }
        std::vector<std::string> lines = _iter_rttm();
        for (const auto& line : lines) {
            file << line;
        }
        file.close();
    }


    std::vector<std::string> _iter_lab() const {
        std::vector<std::string> lines;

        for (const auto& track_item : _tracks) {
            const Segment& segment = track_item.first;
            for (const auto& track : track_item.second) {
                const std::string& label = track.second;

                // Check for spaces in labels if needed
                if (label.find(' ') != std::string::npos) {
                    throw std::runtime_error("Label cannot contain spaces for LAB format.");
                }

                std::ostringstream line;
                line.precision(3);
                line << std::fixed << segment.getStart() << " "
                     << segment.getEnd() << " " << label << "\n";
                lines.push_back(line.str());
            }
        }

        return lines;
    }

    std::string to_lab() const {
        std::vector<std::string> lines = _iter_lab();
        std::ostringstream oss;
        for (const auto& line : lines) {
            oss << line;
        }
        return oss.str();
    }

    void write_lab(const std::string& filename) const {
        std::ofstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open file for writing.");
        }
        std::vector<std::string> lines = _iter_lab();
        for (const auto& line : lines) {
            file << line;
        }
        file.close();
    }

    std::shared_ptr<Annotation> crop(const std::shared_ptr<Timeline>& support, const std::string& mode) {
        auto supportedTimeline = support->support();  // Assuming support() returns a Timeline object

        std::shared_ptr<Annotation> cropped = std::make_shared<Annotation>(_uri, modality);
        const auto& supportSegments = supportedTimeline.getSegments();  // Correct use of object returned by value

        if (mode == "loose") {
            for (const auto& segment : _timeline->getSegments()) {
                for (const auto& supportSegment : supportSegments) {
                    if (segment.intersects(supportSegment)) {
                        cropped->_tracks[segment] = _tracks[segment];
                    }
                }
            }
        } else if (mode == "strict") {
            for (const auto& segment : _timeline->getSegments()) {
                for (const auto& supportSegment : supportSegments) {
                    if (supportSegment.contains(segment)) {
                        cropped->_tracks[segment] = _tracks[segment];
                    }
                }
            }
        } else if (mode == "intersection") {
            for (const auto& segment : _timeline->getSegments()) {
                for (const auto& supportSegment : supportSegments) {
                    if (segment.intersects(supportSegment)) {
                        Segment intersection = segment&supportSegment;
                        cropped->_tracks[intersection] = _tracks[segment];
                    }
                }
            }
        } else {
            throw std::runtime_error("Unsupported mode: " + mode);
        }

        return cropped;
    }


    std::shared_ptr<Annotation> extrude(std::shared_ptr<Timeline>& removed, const std::string& mode) {
        // タイムラインの確認と展開
        if (removed->getSegments().size() == 1) {
            // 単一セグメントをタイムラインに変換
            Segment singleSegment = removed->getSegments()[0];
            removed = std::make_shared<Timeline>(std::vector<Segment>{singleSegment}, removed->getUri());
        }

        // アノテーションの全範囲を取得
        auto currentTimeline = get_timeline(false);  // Shallow copy for internal use
        Timeline extent_tl(std::vector<Segment>{currentTimeline->extent()}, _uri.value_or(""));

        // removed Timeline のギャップを計算
        // Support 型に Timeline のポインタをラップして渡す
        Support supportTimeline = &extent_tl;
        auto truncating_support = std::make_shared<Timeline>(removed->gaps(supportTimeline));

        // モードの調整： "loose" for truncate means "strict" for crop, and vice versa
        std::string cropMode = (mode == "loose") ? "strict" : (mode == "strict") ? "loose" : mode;

        // クロップされたアノテーションを作成し、新しいAnnotationを返す
        return this->crop(truncating_support, cropMode);
    }


    std::shared_ptr<Timeline> get_overlap(const std::vector<Label>& labels = {}) {
        std::shared_ptr<Annotation> filtered = std::make_shared<Annotation>(_uri, modality); // 新しいAnnotationインスタンスを作成
        if (!labels.empty()) {
            filtered = subset(labels); // ラベルでフィルタリング
        }

        auto overlaps = std::make_shared<Timeline>(filtered->getUri().value_or("")); // 重複部分を格納するタイムライン
        auto coIterResults = filtered->co_iter(*filtered);
        for (const auto& pair : coIterResults) {
            const auto& firstTrack = pair.first;
            const auto& secondTrack = pair.second;
            const Segment& firstSegment = std::get<0>(firstTrack);
            const Segment& secondSegment = std::get<0>(secondTrack);
            const TrackName& firstTrackName = std::get<1>(firstTrack);
            const TrackName& secondTrackName = std::get<1>(secondTrack);

            // 重複をチェック（異なるトラック間のみ）
            if (firstSegment != secondSegment && firstTrackName != secondTrackName) {
                overlaps->add(firstSegment & secondSegment); // セグメントの交差部分を追加
            }
        }

        return std::make_shared<Timeline>(overlaps->support());
    }

    std::set<TrackName> get_tracks(const Segment& segment) const {
        std::set<TrackName> trackNames;
        auto it = _tracks.find(segment);
        if (it != _tracks.end()) {
            for (const auto& track : it->second) {
                trackNames.insert(track.first);
            }
        }
        return trackNames;
    }

    bool has_track(const Segment& segment, const TrackName& trackName) const {
        // _tracks マップから segment をキーとして検索
        auto it = _tracks.find(segment);
        if (it != _tracks.end()) {
            // segment が見つかった場合、次に trackName がそのセグメントに存在するかチェック
            const auto& tracks = it->second;  // このセグメントに対応するトラック名とラベルのマップ
            return tracks.find(trackName) != tracks.end();  // trackName が存在するかどうかを返す
        }
        return false;  // segment が見つからない場合は false を返す
    }

    std::shared_ptr<Annotation> copy() const {
        // 新しいAnnotationオブジェクトを作成。URIとモダリティを元のオブジェクトからコピーします。
        auto copied = std::make_shared<Annotation>(_uri, modality);

        // _tracks マップをディープコピーします。
        for (const auto& trackPair : _tracks) {
            Segment segment = trackPair.first;
            std::map<TrackName, Label> tracks = trackPair.second;
            copied->_tracks[segment] = tracks;
        }

        // _labels マップもディープコピーします。
        for (const auto& labelPair : _labels) {
            Label label = labelPair.first;
            if (labelPair.second) {
                // Timelineをディープコピーするために、Timelineのコピーコンストラクタを呼び出します。
                copied->_labels[label] = std::make_shared<Timeline>(*labelPair.second);
            } else {
                copied->_labels[label] = nullptr;
            }
            copied->_labelNeedsUpdate[label] = _labelNeedsUpdate.at(label);
        }

        // タイムラインは必要に応じて更新されるようにマークします。
        copied->_timelineNeedsUpdate = true;

        return copied;
    }

    TrackName new_track(
        const Segment& segment,
        const std::optional<TrackName>& candidate = std::nullopt,
        const std::optional<std::string>& prefix = std::nullopt
    ) {
        // 既存のトラック名を取得
        std::set<TrackName> existing_tracks;
        if (_tracks.find(segment) != _tracks.end()) {
            for (const auto& track : _tracks[segment]) {
                existing_tracks.insert(track.first);
            }
        }

        // 提供された候補が存在し、まだ使われていない場合は、それを使用する
        if (candidate && (existing_tracks.find(candidate.value()) == existing_tracks.end())) {
            return candidate.value();
        }

        // プレフィックスを確認し、未指定の場合は空文字列を使用
        std::string effective_prefix = prefix.value_or("");

        // 新しいトラック名を生成（既存のものと重複しないように）
        int count = 0;
        TrackName new_track_name;
        do {
            new_track_name = effective_prefix + std::to_string(count++);
        } while (existing_tracks.find(new_track_name) != existing_tracks.end());

        // 新しいトラック名を返す
        return new_track_name;
    }

    std::string __str__() const {
        // トラック情報を取得します。ラベルも含めて取得するために、iterTracksをtrueで呼び出します。
        auto tracks = iterTracks(true);

        // 文字列ストリームを使用して、各トラック情報をフォーマットします。
        std::ostringstream stream;
        for (const auto& track : tracks) {
            const Segment& segment = std::get<0>(track);
            const TrackName& trackName = std::get<1>(track);
            const std::optional<Label>& label = std::get<2>(track);

            // ラベルが存在する場合はその値を、そうでなければ"N/A"を使用します。
            std::string label_str = label ? *label : "N/A";

            // Segmentの文字列表現を取得
            std::string segment_str = segment.__str__();

            // TrackNameの値に応じた文字列を取得
            std::string trackName_str = std::visit([](auto&& arg) -> std::string {
                using T = std::decay_t<decltype(arg)>;
                if constexpr (std::is_same_v<T, std::string>) {
                    return arg;  // argがstd::stringの場合
                } else {
                    return std::to_string(arg);  // argがintの場合
                }
            }, trackName);

            // セグメント、トラック名、ラベルをフォーマットして追加します。
            stream << segment_str << " " << trackName_str << " " << label_str << "\n";
        }

        // 生成した文字列を返します。
        return stream.str();
    }

    void __delitem__(const Segment& segment) {
        auto it = _tracks.find(segment);
        if (it != _tracks.end()) {
            // トラックに関連付けられた各ラベルの更新フラグを設定
            for (const auto& track : it->second) {
                _labelNeedsUpdate[track.second] = true;
            }

            // セグメントを_tracksから削除
            _tracks.erase(it);

            // タイムラインの更新が必要
            _timelineNeedsUpdate = true;
        } else {
            throw std::runtime_error("Segment not found.");
        }
    }

    void __delitem__(const Segment& segment, const TrackName& trackName) {
        auto it = _tracks.find(segment);
        if (it != _tracks.end()) {
            auto& tracks = it->second;
            auto trackIt = tracks.find(trackName);

            if (trackIt != tracks.end()) {
                // ラベルの更新フラグを設定
                _labelNeedsUpdate[trackIt->second] = true;

                // トラックを削除
                tracks.erase(trackIt);

                // トラックのコレクションが空になった場合はセグメントも削除
                if (tracks.empty()) {
                    _tracks.erase(it);
                    _timelineNeedsUpdate = true;
                }
            } else {
                throw std::runtime_error("Track not found.");
            }
        } else {
            throw std::runtime_error("Segment not found.");
        }
    }

    Label __getitem__(const Segment& segment, const TrackName& trackName = "_") const {
        auto segIt = _tracks.find(segment);
        if (segIt != _tracks.end()) {
            const auto& tracks = segIt->second;
            auto trackIt = tracks.find(trackName);
            if (trackIt != tracks.end()) {
                return trackIt->second;
            } else {
                throw std::runtime_error("Track not found.");
            }
        } else {
            throw std::runtime_error("Segment not found.");
        }
    }

    void __setitem__(const Segment& segment, const TrackName& trackName, const Label& label) {
        if (!segment) {
            // セグメントが空の場合は何もしない
            return;
        }

        // セグメントが_tracksに存在しない場合は、新しいセグメントとして追加
        auto& tracks = _tracks[segment];
        if (tracks.find(trackName) == tracks.end()) {
            // 新しいトラックを追加する場合、タイムラインを更新する必要があります
            _timelineNeedsUpdate = true;
        } else {
            // 既存のトラックを更新する場合、古いラベルの更新が必要です
            Label& oldLabel = tracks[trackName];
            if (oldLabel != label) {
                _labelNeedsUpdate[oldLabel] = true;
            }
        }

        // 新しいラベルを設定
        tracks[trackName] = label;

        // 新しいラベルに対しても更新フラグを設定
        _labelNeedsUpdate[label] = true;
    }

    std::shared_ptr<Annotation> empty() {
        // 新しい空のAnnotationオブジェクトを生成し、現在のオブジェクトのURIとモダリティを使用します。
        return std::make_shared<Annotation>(_uri, modality);
    }

    std::vector<Label> labels() {
        // ラベルが更新されているかどうかを確認し、必要があれば更新を行います。
        if (std::any_of(_labelNeedsUpdate.begin(), _labelNeedsUpdate.end(),
                        [](const std::pair<Label, bool>& lnu) { return lnu.second; })) {
            updateLabels();
        }
        
        // ラベルのリストを作成し、ソートして返します。
        std::vector<Label> sortedLabels;
        for (const auto& labelPair : _labels) {
            sortedLabels.push_back(labelPair.first);
        }
        std::sort(sortedLabels.begin(), sortedLabels.end());
        return sortedLabels;
    }

    std::vector<Label> getLabels() {
        // ラベルが更新されているかどうかを確認し、必要ならば更新を実施
        for (const auto& label_update : _labelNeedsUpdate) {
            if (label_update.second) {
                updateLabels();
                break; // 更新が必要なラベルがあれば、updateLabelsを呼び出して抜ける
            }
        }

        // ラベルだけを抽出してvectorに格納
        std::vector<Label> labels;
        for (const auto& label : _labels) {
            labels.push_back(label.first);
        }

        // ラベルをソート（Label型がソート可能である必要がある）
        std::sort(labels.begin(), labels.end());

        return labels;
    }

    std::shared_ptr<Annotation> subset(const std::vector<Label>& filterLabels, bool invert = false) {
        // ベクターからセットへの変換を明示的に行う
        std::set<Label> labelSet(filterLabels.begin(), filterLabels.end());

        // getLabels() から返されたベクターをセットに変換
        std::vector<Label> existingLabelsVec = getLabels();
        std::set<Label> existingLabels(existingLabelsVec.begin(), existingLabelsVec.end());

        // インバートフラグが設定されている場合、フィルタリングするラベルを反転させる
        if (invert) {
            std::set<Label> temp;
            std::set_difference(existingLabels.begin(), existingLabels.end(),
                                labelSet.begin(), labelSet.end(),
                                std::inserter(temp, temp.begin()));
            labelSet = temp;
        } else {
            std::set<Label> temp;
            std::set_intersection(labelSet.begin(), labelSet.end(),
                                existingLabels.begin(), existingLabels.end(),
                                std::inserter(temp, temp.begin()));
            labelSet = temp;
        }

        // 新しいAnnotationオブジェクトを作成
        std::shared_ptr<Annotation> subsetAnnotation = std::make_shared<Annotation>(_uri, modality);

        // ラベルに基づいてトラックをフィルタリング
        for (const auto& trackItem : _tracks) {
            const Segment& segment = trackItem.first;
            std::map<TrackName, Label> filteredTracks;

            for (const auto& track : trackItem.second) {
                if (labelSet.find(track.second) != labelSet.end()) {
                    filteredTracks[track.first] = track.second;
                }
            }

            if (!filteredTracks.empty()) {
                subsetAnnotation->_tracks[segment] = filteredTracks;
            }
        }

        // 必要なメンバ変数を更新
        subsetAnnotation->_labelNeedsUpdate = _labelNeedsUpdate;
        subsetAnnotation->_labels = _labels;
        subsetAnnotation->_timelineNeedsUpdate = true;

        return subsetAnnotation;
    }

    std::shared_ptr<Annotation> update(const std::shared_ptr<Annotation>& annotation, bool copy = false) {
        // コピーが必要かどうかに応じて、現在のオブジェクトのコピーを作成するか、自分自身を使用する
        std::shared_ptr<Annotation> result;
        if (copy) {
            result = this->copy();  // コピーを作成
        } else {
            result = std::make_shared<Annotation>(*this);  // 新しい shared_ptr を作成し、現在のオブジェクトの内容をコピー
        }

        // 別のアノテーションから全てのトラックを取得し、現在のアノテーションに追加または更新
        auto tracks = annotation->iterTracks(true);  // ラベル情報も含めてトラックを取得
        for (const auto& track : tracks) {
            const Segment& segment = std::get<0>(track);
            const TrackName& trackName = std::get<1>(track);
            const Label& label = std::get<2>(track).value();  // ラベル情報を取得

            // セグメントとトラック名を指定してラベルを設定または更新
            result->__setitem__(segment, trackName, label);
        }

        return result;
    }

    std::shared_ptr<Timeline> label_timeline(const Label& label, bool copy) {
        // ラベルが存在しなければ、新しい空のタイムラインを作成
        if (_labels.find(label) == _labels.end()) {
            return std::make_shared<Timeline>(_uri.value_or(""));
        }

        // ラベルのデータが更新が必要な場合は更新を行う
        if (_labelNeedsUpdate[label]) {
            updateLabels();
        }

        // コピーが要求された場合は、タイムラインのコピーを返す
        if (copy) {
            return std::make_shared<Timeline>(*_labels[label]);
        } else {
            // コピーが不要な場合は、直接参照を返す
            return _labels[label];
        }
    }

    Timeline label_support(const Label& label) {
        // label_timeline メソッドを使用して、コピーせずにタイムラインを取得し、そのサポートを取得
        auto timeline = label_timeline(label, false);
        return timeline->support();
    }

    double label_duration(const Label& label) {
        // label_timeline メソッドを使用して、コピーせずにタイムラインを取得し、その持続時間を取得
        auto timeline = label_timeline(label, false);
        return timeline->duration();
    }

    std::vector<std::pair<Label, float>> chart(bool percent = false) {
        std::vector<std::pair<Label, float>> chart;
        auto labels = this->labels(); // すでに実装されているlabels()関数を使用
        float totalDuration = 0.0;

        // 各ラベルの持続時間を計算
        for (const auto& label : labels) {
            float duration = this->label_duration(label);
            chart.emplace_back(label, duration);
            totalDuration += duration;
        }

        // 持続時間で降順にソート
        std::sort(chart.begin(), chart.end(), [](const std::pair<Label, float>& a, const std::pair<Label, float>& b) {
            return a.second > b.second;
        });

        // パーセント表示が必要な場合は持続時間をパーセントに変換
        if (percent && totalDuration > 0) {
            for (auto& item : chart) {
                item.second = (item.second / totalDuration) * 100.0;
            }
        }

        return chart;
    }

    std::optional<Label> argmax(std::shared_ptr<Timeline> support = nullptr) {
        std::shared_ptr<Annotation> cropped;

        // サポートが提供されている場合は、そのサポートに基づいてアノテーションをクロップ
        if (support) {
            cropped = this->crop(support, "intersection");
        } else {
            // サポートがない場合は、元のアノテーションのコピーを作成
            cropped = this->copy();
        }

        // クロップされたアノテーションが空の場合は何も返さない
        if (!cropped || cropped->__len__() == 0) {
            return std::nullopt;
        }

        // 最も長い持続時間を持つラベルを探す
        std::optional<Label> maxLabel;
        float maxDuration = 0.0;
        for (const auto& label : cropped->labels()) {
            float duration = cropped->label_duration(label);
            if (duration > maxDuration) {
                maxDuration = duration;
                maxLabel = label;
            }
        }

        return maxLabel;
    }

    std::shared_ptr<Annotation> rename_tracks(const std::string& generator_type) {
        auto new_annotation = std::make_shared<Annotation>(_uri, modality);  // 新しいアノテーションオブジェクトを作成
        std::function<std::string()> generator; // 名前ジェネレータ

        if (generator_type == "string") {
            generator = string_generator();
        } else if (generator_type == "int") {
            auto int_gen = int_generator();
            generator = [&int_gen]() { return std::to_string(int_gen()); };
        }

        // 全てのトラックを新しい名前で再割り当て
        for (auto& track_item : _tracks) {
            const Segment& segment = track_item.first;
            for (auto& track : track_item.second) {
                new_annotation->_tracks[segment][generator()] = track.second; // 新しいトラック名を割り当て
            }
        }

        return new_annotation;
    }

    std::shared_ptr<Annotation> rename_labels(const std::map<std::string, std::string>& mapping, const std::string& generator_type, bool copy) {
        std::shared_ptr<Annotation> result_annotation = copy ? this->copy() : std::make_shared<Annotation>(*this);  // コピーを作成するか元のインスタンスを使う

        std::function<std::string()> generator; // 名前ジェネレータ

        if (generator_type == "string") {
            generator = string_generator();
        } else if (generator_type == "int") {
            auto int_gen = int_generator();
            generator = [&int_gen]() { return std::to_string(int_gen()); };
        }

        std::map<std::string, std::string> effective_mapping = mapping;

        if (mapping.empty()) {
            // マッピングが提供されていない場合、すべてのラベルに対して新しいラベル名を生成
            for (const auto& label : this->labels()) {
                effective_mapping[label] = generator();
            }
        }

        // ラベル名を更新
        for (auto& track_item : result_annotation->_tracks) {
            std::map<TrackName, Label> new_track_labels;

            for (auto& track : track_item.second) {
                std::string new_label = (effective_mapping.find(track.second) != effective_mapping.end()) ? effective_mapping[track.second] : track.second;
                new_track_labels[track.first] = new_label;
            }

            track_item.second = new_track_labels;
        }

        // ラベル更新フラグをリセット
        for (auto& label : result_annotation->_labelNeedsUpdate) {
            label.second = true;  // 新しいラベルに更新フラグを設定
        }

        return result_annotation;
    }


    std::shared_ptr<Annotation> relabel_tracks(const std::string& generator_type) {
        auto new_annotation = std::make_shared<Annotation>(_uri, modality);  // 新しいアノテーションオブジェクトを作成
        std::function<std::string()> generator; // 名前ジェネレータ

        if (generator_type == "string") {
            generator = string_generator();
        } else if (generator_type == "int") {
            auto int_gen = int_generator();
            generator = [&int_gen]() { return std::to_string(int_gen()); };
        }

        // 全てのトラックにユニークなラベルを割り当て
        for (auto& track_item : _tracks) {
            const Segment& segment = track_item.first;
            for (auto& track : track_item.second) {
                new_annotation->_tracks[segment][track.first] = generator(); // 新しいラベルを割り当て
            }
        }

        return new_annotation;
    }

    std::shared_ptr<Annotation> support(double collar = 0.0) {
        // 新しいAnnotationオブジェクトを作成（URIとmodalityは継承）
        auto supportAnnotation = std::make_shared<Annotation>(_uri, modality);

        // ラベルごとに処理
        for (const auto& label_pair : _labels) {
            auto label = label_pair.first;
            auto timeline = label_pair.second;

            // タイムラインのサポートを計算
            auto supportedTimeline = timeline->support(collar);

            // サポートされたタイムラインのセグメントを新しいAnnotationに追加
            for (const auto& segment : supportedTimeline.getSegments()) {
                // ラベルとトラック名の生成（トラック名は省略可能）
                std::string trackName = "merged";  // 例: 一意のトラック名生成は省略
                supportAnnotation->_tracks[segment][trackName] = label;
            }
        }

        // タイムラインの更新が必要
        supportAnnotation->_timelineNeedsUpdate = true;

        return supportAnnotation;
    }

    std::vector<std::pair<std::tuple<Segment, TrackName>, std::tuple<Segment, TrackName>>> co_iter(Annotation& other) {
        std::vector<std::pair<std::tuple<Segment, TrackName>, std::tuple<Segment, TrackName>>> result;

        // 両方のアノテーションのタイムラインを取得
        auto thisTimeline = get_timeline(false);
        auto otherTimeline = other.get_timeline(false);

        // タイムラインの co_iter を利用して交差するセグメントのペアを取得
        auto intersections = thisTimeline->co_iter(*otherTimeline);

        // 交差する各セグメントペアに対してトラックの組み合わせを見つける
        for (const auto& intersection : intersections) {
            const Segment& thisSegment = intersection.first;
            const Segment& otherSegment = intersection.second;

            auto thisTracks = get_tracks(thisSegment);
            auto otherTracks = other.get_tracks(otherSegment);

            // すべてのトラック名の組み合わせを返す
            for (const auto& thisTrack : thisTracks) {
                for (const auto& otherTrack : otherTracks) {
                    result.emplace_back(std::make_tuple(thisSegment, thisTrack),
                                        std::make_tuple(otherSegment, otherTrack));
                }
            }
        }

        return result;
    }

    std::vector<std::vector<float>> __mul__(Annotation& other) {
        std::map<Label, int> i_labels, j_labels;
        int i = 0, j = 0;
        
        // Indexing labels
        for (const auto& label : labels()) {
            i_labels[label] = i++;
        }
        for (const auto& label : other.labels()) {
            j_labels[label] = j++;
        }

        // Initialize the matrix
        std::vector<std::vector<float>> matrix(i_labels.size(), std::vector<float>(j_labels.size(), 0.0f));

        // Calculate intersections and durations
        auto co_occurrences = co_iter(other);
        for (const auto& co : co_iter(other)) {
            const auto& seg1 = std::get<0>(co.first);
            const auto& seg2 = std::get<0>(co.second);
            auto label1 = __getitem__(seg1, std::get<1>(co.first));
            auto label2 = other.__getitem__(seg2, std::get<1>(co.second));
            float duration = (seg1 & seg2).duration();  // Assuming Segment has a duration() method that calculates the intersection duration
            matrix[i_labels[label1]][j_labels[label2]] += duration;
        }


        return matrix;
    }


    // Annotationクラス内での実装
    std::shared_ptr<SlidingWindowFeature> discretize(
        const std::optional<Segment>& support,
        double resolution,
        const std::vector<Label>& labels,
        const std::optional<double>& duration
    ) {
        Segment actualSupport = support.value_or(get_timeline()->extent());
        double startTime = actualSupport.getStart();
        double endTime = actualSupport.getEnd();

        // クロップしたアノテーションを取得
        auto cropped = crop(std::make_shared<Timeline>(std::vector<Segment>{actualSupport}), "intersection");

        std::vector<Label> actualLabels = labels.empty() ? this->labels() : labels;

        SlidingWindow resolutionWindow(startTime, resolution, resolution, std::numeric_limits<double>::infinity());
        
        int startFrame = resolutionWindow.closest_frame(startTime);
        int endFrame = duration.has_value() ? startFrame + static_cast<int>(std::ceil(duration.value() / resolution)) : resolutionWindow.closest_frame(endTime);
        int numFrames = endFrame - startFrame;

        std::vector<std::vector<double>> data(numFrames, std::vector<double>(actualLabels.size(), 0));

        for (size_t k = 0; k < actualLabels.size(); ++k) {
            auto timeline = cropped->label_timeline(actualLabels[k], false); // タイムラインを取得
            const auto& segments = timeline->getSegments(); // タイムラインからセグメントのベクターを取得
            for (const auto& segment : segments) { // セグメントのベクターをイテレート
                auto cropResult = resolutionWindow.crop(segment, "center", true);
                if (cropResult.size() >= 2) { // cropResultが少なくとも2つの要素を含むことを確認
                    int start = cropResult[0];
                    int end = cropResult[1];
                    for (int i = std::max(0, start); i < std::min(end, numFrames); ++i) {
                        data[i][k] = 1;
                    }
                }
            }
        }

        // Ensure data is binary
        for (auto& row : data) {
            for (auto& value : row) {
                value = std::min(value, 1.0);
            }
        }

        return std::make_shared<SlidingWindowFeature>(data, resolutionWindow, actualLabels);
    }


    static std::shared_ptr<Annotation> fromRecords(
        const std::vector<std::tuple<Segment, TrackName, Label>>& records,
        const std::optional<std::string>& uri = std::nullopt,
        const std::optional<std::string>& modality = std::nullopt
    ) {
        auto annotation = std::make_shared<Annotation>(uri, modality);
        std::set<Label> labels;

        for (const auto& record : records) {
            const auto& [segment, track, label] = record;
            annotation->_tracks[segment][track] = label;
            labels.insert(label);
        }

        for (const auto& label : labels) {
            annotation->_labels[label] = nullptr; // Timeline instances to be created later as needed
            annotation->_labelNeedsUpdate[label] = true;
        }
        annotation->_timelineNeedsUpdate = true;

        return annotation;
    }

    // _repr_png_ は colab用なのでなし


};