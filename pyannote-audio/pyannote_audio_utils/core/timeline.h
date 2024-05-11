#ifndef TIMELINE_H
#define TIMELINE_H

#include <vector>
#include <set>
#include <algorithm>
#include <string>
#include <optional>
#include <variant>

class Timeline {
private:
    std::vector<Segment> segmentsVector;
    std::set<Segment> segmentsSet;
    std::string uri;

public:
    Timeline(const std::string& uri);
    Timeline(const std::vector<Segment>& segments = {}, const std::string& uri = "");

    void setUri(const std::string& newUri);
    const std::vector<Segment>& getSegments() const;
    size_t __len__() const;
    operator bool() const;
    std::vector<Segment>::const_iterator begin() const;
    std::vector<Segment>::const_iterator end() const;
    const Segment& operator[](size_t index) const;
    bool operator==(const Timeline& other) const;
    bool operator!=(const Timeline& other) const;
    int index(const Segment& segment) const;
    void add(const Segment& segment);
    void remove(const Segment& segment);
    void discard(const Segment& segment);
    Timeline& operator|=(const Timeline& other);
    Timeline& update(const Timeline& other);
    Timeline operator|(const Timeline& other) const;
    Timeline unionWith(const Timeline& other) const;
    std::vector<std::pair<Segment, Segment>> co_iter(const Timeline& other) const;
    std::vector<Segment> crop_iter(const Support& support, const std::string& mode = "intersection") const;
    Timeline crop(const Support& support, const std::string& mode = "intersection") const;
    std::vector<Segment> overlapping(double t) const;
    Timeline get_overlap() const;
    Timeline extrude(const Timeline& removed, const std::string& mode = "intersection") const;
    std::string __str__() const;
    std::string __repr__() const;
    bool __contains__(const Segment& segment) const;
    bool __contains__(const Timeline& timeline) const;
    Timeline empty() const;
    bool covers(const Timeline& other) const;
    Timeline copy(std::function<Segment(Segment)> segmentFunc = nullptr) const;
    Segment extent() const;
    std::vector<Segment> support_iter(double collar = 0.0) const;
    Timeline support(double collar = 0.0) const;
    double duration() const;
    std::vector<Segment> gaps_iter(const Support& support) const;
    Timeline gaps(const Support& support = {}) const;
    Timeline segmentation() const;
    // Annotation toAnnotation(const std::string& generatorType, const std::string& modality) const;
    std::string to_uem() const;
    void write_uem(const std::string& filename) const;
};

#endif // TIMELINE_H
