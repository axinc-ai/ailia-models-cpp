#include <vector>
#include <string>
#include <optional>
#include <map>

#include "types.h"
#include "segment.h"

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
    // std::map<Label, std::shared_ptr<Timeline>> _labels;
    // std::map<Label, bool> _labelNeedsUpdate;
    // std::shared_ptr<Timeline> _timeline;
    // bool _timelineNeedsUpdate;


    Annotation fromRecords( //from_dfと対応
    const std::vector<Record>& records,
    const std::optional<std::string>& uri,
    const std::optional<std::string>& modality
    ) {
        // RecordsをもとにAnnotationオブジェクトを構築するロジックをここに実装
        // この例では、単純にレコードリストを受け取り、Annotationオブジェクトを返すとしています。
        Annotation annotation;
        // Annotationの初期化や設定を行う
        return annotation;
    }

    //コンストラクタ
    // Annotation::Annotation(const std::optional<std::string>& uri, const std::optional<std::string>& modality)
    // : _uri(uri), modality(modality), _timelineNeedsUpdate(true) {
    // _timeline = std::make_shared<Timeline>();
};