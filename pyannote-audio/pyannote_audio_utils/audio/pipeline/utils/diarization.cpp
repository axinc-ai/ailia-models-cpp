#include <optional>
#include <limits>
#include <stdexcept>
#include <tuple>
#include <string>


#include "annotation.h"


int diarization_test(int input)
{
    return input^3;
}



class SpeakerDiarizationMixin
{
    static std::tuple<std::optional<int>, int, std::optional<int>> set_num_speakers(
        std::optional<int> num_speakers = std::nullopt,
        std::optional<int> min_speakers = std::nullopt,
        std::optional<int> max_speakers = std::nullopt
    ) {
        // Define default values
        int resolved_min_speakers = min_speakers.value_or(1);
        int resolved_max_speakers = max_speakers.value_or(std::numeric_limits<int>::max());

        // Override min and max if num_speakers is specified
        if (num_speakers.has_value()) {
            resolved_min_speakers = num_speakers.value();
            resolved_max_speakers = num_speakers.value();
        }

        // Check consistency
        if (resolved_min_speakers > resolved_max_speakers) {
            throw std::invalid_argument(
                "min_speakers must be smaller than or equal to max_speakers " +
                std::string("(here: min_speakers=") + std::to_string(resolved_min_speakers) +
                " and max_speakers=" + std::to_string(resolved_max_speakers) + ")."
            );
        }

        // If min and max speakers are the same, we resolve num_speakers to their value
        if (resolved_min_speakers == resolved_max_speakers) {
            num_speakers = resolved_min_speakers;
        }

        return {num_speakers, resolved_min_speakers, resolved_max_speakers};
    }

};