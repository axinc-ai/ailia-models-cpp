/*******************************************************************
*
*    DESCRIPTION:
*      AILIA G2P EN expand
*    AUTHOR:
*
*    DATE:2024/06/26
*
*******************************************************************/

#include <iostream>
#include <regex>
#include <string>
#include <algorithm>
#include <map>
#include <sstream>

using namespace std;

namespace ailiaG2P{

string remove_commas(const smatch &match) {
	string result = match.str(1);
	result.erase(remove(result.begin(), result.end(), ','), result.end());
	return result;
}

string expand_decimal_point(const smatch &match) {
	string result = match.str(1);
	size_t index = result.find('.');
	if (index != string::npos) {
		result.replace(index, 1, " point ");
	}
	return result;
}

string expand_dollars(const smatch &match) {
	string match_str = match.str(1);
	size_t dot_pos = match_str.find('.');

	int dollars = 0;
	int cents = 0;

	if (dot_pos != string::npos) {
		dollars = stoi(match_str.substr(0, dot_pos));
		if (dot_pos + 1 < match_str.size()) {
			cents = stoi(match_str.substr(dot_pos + 1));
		}
	} else {
		dollars = stoi(match_str);
	}

	string result;
	if (dollars > 0) {
		result += to_string(dollars) + " dollar" + (dollars == 1 ? "" : "s");
	}

	if (cents > 0) {
		if (!result.empty()) {
			result += ", ";
		}
		result += to_string(cents) + " cent" + (cents == 1 ? "" : "s");
	}

	if (result.empty()) {
		result = "zero dollars";
	}

	return result;
}

string expand_ordinal(const smatch &match) {
	static const map<string, string> special_cases = {
		{"1st", "first"}, {"2nd", "second"}, {"3rd", "third"}, {"4th", "fourth"},
		{"5th", "fifth"}, {"6th", "sixth"}, {"7th", "seventh"}, {"8th", "eighth"},
		{"9th", "ninth"}, {"10th", "tenth"}};

	string num_str = match.str(0);
	if (special_cases.find(num_str) != special_cases.end()) {
		return special_cases.at(num_str);
	}
	return num_str;
}

string number_to_words(int num) {
	if (num == 0) {
		return "zero";
	}

	static const string units[] = {
		"", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
		"eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"
	};

	static const string tens[] = {
		"", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"
	};

	static const string thousands[] = {
		"", "thousand", "million", "billion"
	};

	string result;
	int place = 0;

	while (num > 0) {
		int n = num % 1000;
		if (n != 0) {
			string segment;
			if (n >= 100) {
				segment += units[n / 100] + " hundred ";
				n %= 100;
			}
			if (n >= 20) {
				segment += tens[n / 10] + " ";
				n %= 10;
			}
			if (n > 0) {
				segment += units[n] + " ";
			}
			result = segment + thousands[place] + " " + result;
		}
		place++;
		num /= 1000;
	}

	// Remove any extra spaces at the end
	result.erase(result.find_last_not_of(" \n\r\t") + 1);

	return result;
}

string expand_number(const smatch &match) {
	int num = stoi(match.str(0));
	return number_to_words(num);
}

string normalize_numbers(const string &text) {
	string result = text;

	regex comma_number_re(R"((\d[\d\,]*\d))");
	regex decimal_number_re(R"((\d+\.\d+))");
	regex pounds_re(R"(£([\d\,]*\d))");
	regex dollars_re(R"(\$([\d\.\,]*\d))");
	regex ordinal_re(R"(\d+(st|nd|rd|th))");
	regex number_re(R"(\d+)");

	smatch match;

	auto update_result = [&](const regex& re, function<string(const smatch&)> fn) {
		string res;
		auto begin = result.cbegin();
		auto end = result.cend();
		while (regex_search(begin, end, match, re)) {
			res.append(begin, match[0].first);
			res.append(fn(match));
			begin = match[0].second;
		}
		res.append(begin, end);
		result.swap(res);
	};

	update_result(comma_number_re, remove_commas);
	update_result(pounds_re, [](const smatch &m) { return m.str(1) + " pounds"; });
	update_result(decimal_number_re, expand_decimal_point);
	update_result(dollars_re, expand_dollars);
	update_result(ordinal_re, expand_ordinal);
	update_result(number_re, expand_number);

	return result;
}

void test_expand() {
	string text = "I have £1,000 and $1,234.56 and this is my 1st test.";
	string output = normalize_numbers(text);
	string expect = "I have one thousand pounds and one thousand two hundred thirty four dollars point fifty six and this is my first test.";
	if (output != expect){
		throw("verify error at test_expand");
	}
}

}
