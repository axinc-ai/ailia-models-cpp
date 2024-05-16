#ifndef ANNOTATION_H
#define ANNOTATION_H

#include <vector>
#include <string>
#include <optional>
#include <map>
#include <set>
#include <memory>
#include <fstream>
#include <sstream>
#include <iostream>
#include <functional>
#include <algorithm>
#include <stdexcept>
#include <cmath>
#include <limits>

#include "types.h"
// #include "segment.h"
// #include "timeline.h"
// #include "generators.h"
// #include "feature.h"

struct Record {
    double start;
    double end;
    int track;
    std::string label;
};

class Annotation {
private:
    std::optional<std::string> _uri;
    std::optional<std::string> modality;
    std::map<Segment, std::map<TrackName, Label>> _tracks;
    std::map<Label, std::shared_ptr<Timeline>> _labels;
    std::map<Label, bool> _labelNeedsUpdate;
    std::shared_ptr<Timeline> _timeline;
    bool _timelineNeedsUpdate;
public:
    Annotation(const std::optional<std::string>& uri = std::nullopt, const std::optional<std::string>& modality = std::nullopt);
    std::optional<std::string> getUri() const;
    void setUri(const std::string& uri);
    void updateLabels();
    size_t __len__() const;
    bool __bool__() const;
    std::vector<Segment> itersegments() const;
    std::vector<std::tuple<Segment, TrackName, std::optional<Label>>> iterTracks(bool yield_label = false) const;
    void _updateTimeline();
    std::shared_ptr<Timeline> get_timeline(bool makeCopy = true);
    bool operator==(const Annotation& other) const;
    bool operator!=(const Annotation& other) const;
    bool __contains__(const Segment& segment);
    bool __contains__(const Timeline& included);
    std::vector<std::string> _iter_rttm() const;
    std::string to_rttm() const;
    void write_rttm(const std::string& filename) const;
    std::vector<std::string> _iter_lab() const;
    std::string to_lab() const;
    void write_lab(const std::string& filename) const;
    std::shared_ptr<Annotation> crop(const std::shared_ptr<Timeline>& support, const std::string& mode);
    std::shared_ptr<Annotation> extrude(std::shared_ptr<Timeline>& removed, const std::string& mode);
    std::shared_ptr<Timeline> get_overlap(const std::vector<Label>& labels = {});
    std::set<TrackName> get_tracks(const Segment& segment) const;
    bool has_track(const Segment& segment, TrackName& trackName) const;
    std::shared_ptr<Annotation> copy() const;
    TrackName new_track(const Segment& segment, const std::optional<TrackName>& candidate = std::nullopt, const std::optional<std::string>& prefix = std::nullopt);
    std::string __str__() const;
    void __delitem__(const Segment& segment);
    void __delitem__(const Segment& segment, const TrackName& trackName);
    Label __getitem__(const Segment& segment, const TrackName& trackName = "_") const;
    void __setitem__(const Segment& segment, TrackName& trackName, Label& label);
    std::shared_ptr<Annotation> empty();
    std::vector<Label> labels();
    std::vector<Label> getLabels();
    std::shared_ptr<Annotation> subset(const std::vector<Label>& filterLabels, bool invert = false);
    std::shared_ptr<Annotation> update(const std::shared_ptr<Annotation>& annotation, bool copy = false);
    std::shared_ptr<Timeline> label_timeline(const Label& label, bool copy);
    Timeline label_support(const Label& label);
    double label_duration(const Label& label);
    std::vector<std::pair<Label, float>> chart(bool percent = false);
    std::optional<Label> argmax(std::shared_ptr<Timeline> support = nullptr);
    std::shared_ptr<Annotation> rename_tracks(const std::string& generator_type);
    std::shared_ptr<Annotation> rename_labels(const std::map<std::string, std::string>& mapping, const std::string& generator_type, bool copy);
    std::shared_ptr<Annotation> relabel_tracks(const std::string& generator_type);
    std::shared_ptr<Annotation> support(double collar = 0.0);
    std::vector<std::pair<std::tuple<Segment, TrackName>, std::tuple<Segment, TrackName>>> co_iter(Annotation& other);
    std::vector<std::vector<float>> __mul__(Annotation& other);
    std::shared_ptr<SlidingWindowFeature> discretize(const std::optional<Segment>& support, double resolution, const std::vector<Label>& labels, std::optional<double>& duration);
    static std::shared_ptr<Annotation> fromRecords(const std::vector<std::tuple<Segment, TrackName, Label>>& records, const std::optional<std::string>& uri = std::nullopt, const std::optional<std::string>& modality = std::nullopt);

};


#endif