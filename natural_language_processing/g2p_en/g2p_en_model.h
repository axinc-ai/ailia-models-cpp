/*******************************************************************
*
*    DESCRIPTION:
*      AILIA G2P EN model
*    AUTHOR:
*
*    DATE:2024/06/26
*
*******************************************************************/

#include <vector>
#include "ailia.h"

namespace ailiaG2P{

class G2PEnModel{
private:
	static const int MODEL_N = 2;

	static const int MODEL_ENCODER = 0;
	static const int MODEL_DECODER = 1;

	AILIANetwork* net[MODEL_N];

	std::vector<std::string> predict(const std::string &word);

public:
	int open(int env_id);
	void close(void);
	int compute(std::string text, std::vector<std::string> expect);
};

}