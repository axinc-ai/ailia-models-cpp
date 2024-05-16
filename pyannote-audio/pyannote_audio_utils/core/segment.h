#ifndef SEGMENT_H
#define SEGMENT_H

#include <iostream>
#include <cmath>
#include <vector>
#include <algorithm>
#include <sstream>
#include <iomanip>
#include <stdexcept>
#include <limits>

class Segment {
private:
    double start;
    double end;
    static double precision;
    static bool auto_round;

public:
    Segment(double start = 0.0, double end = 0.0, double precision = 1e-6, bool auto_round = false);
    double getStart() const;
    double getEnd() const;
    bool operator==(const Segment& other) const;
    bool operator!=(const Segment& other) const;
    bool operator<(const Segment& other) const;
    friend std::ostream& operator<<(std::ostream& os, const Segment& segment);
    static void set_precision(int ndigits);
    operator bool() const;
    void __post_init__();
    double duration() const;
    double middle() const;
    Segment copy() const;
    bool contains(const Segment& other) const;
    Segment operator&(const Segment& other) const;
    bool intersects(const Segment& other) const;
    bool overlaps(double t) const;
    Segment operator|(const Segment& other) const;
    Segment operator^(const Segment& other) const;
    std::string __str__() const;
    std::string __repr__() const;
    std::vector<double>::iterator __iter_begin__();
    std::vector<double>::iterator __iter_end__();
};

class SlidingWindow {
private:
    double duration;
    double step;
    double start;
    double end;
    int index;

public:
    SlidingWindow(double duration = 0.030, double step = 0.010, double start = 0.000, double end = std::numeric_limits<double>::infinity());
    double getStart() const;
    double getEnd() const;
    double getStep() const;
    double getDuration() const;
    int closest_frame(double t) const;
    int samples(double from_duration, const std::string& mode = "strict") const;
    std::vector<int> crop(const Segment& focus, const std::string& mode = "loose", double fixed = 0) const;
    std::pair<int, int> segment_to_range(const Segment& segment) const;
    std::pair<int, int> segmentToRange(const Segment& segment) const;
    Segment range_to_segment(int i0, int n) const;
    Segment rangeToSegment(int i0, int n) const;
    double samples_to_duration(int n_samples) const;
    double samplesToDuration(int n_samples) const;
    int duration_to_samples(double duration) const;
    int durationToSamples(double duration) const;
    // Segment __getitem__(int i) const;
    Segment& operator[](int i);
    Segment next();
    SlidingWindow& __iter_begin__();
    SlidingWindow& __iter_end__();
    int __len__() const;
    SlidingWindow copy() const;
    std::vector<Segment> operator()(const Segment& support, bool align_last = false);
};


#endif // SEGMENT_H