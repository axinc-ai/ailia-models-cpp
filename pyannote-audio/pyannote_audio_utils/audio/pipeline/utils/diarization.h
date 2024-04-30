#ifndef _DIARIZATION_H_
#define _DIARIZATION_H_

#include <optional>
#include <limits>
#include <stdexcept>
#include <tuple>
#include <string>



int diarization_test(int input);


class SpeakerDiarizationMixin
{
public:
    /**
     * @brief Validate number of speakers.
     *
     * @param num_speakers int, optional. Number of speakers.
     * @param min_speakers int, optional. Minimum number of speakers.
     * @param max_speakers int, optional. Maximum number of speakers.
     * @return A tuple of (num_speakers, min_speakers, and max_speakers).
     *          num_speakers : int or None
     *          min_speakers : int
     *          max_speakers : int or np.inf
     * @throws std::invalid_argument If min_speakers is greater than max_speakers.
     */
    static std::tuple<std::optional<int>, int, std::optional<int>> set_num_speakers(
        std::optional<int> num_speakers = std::nullopt,
        std::optional<int> min_speakers = std::nullopt,
        std::optional<int> max_speakers = std::nullopt
    );

};


#endif

