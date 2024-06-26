/*******************************************************************
*
*    DESCRIPTION:
*      AILIA G2P EN averaged perceptron
*    AUTHOR:
*
*    DATE:2024/06/26
*
*******************************************************************/


#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <string>
#include <algorithm>

using namespace std;

class AveragedPerceptron {
public:
    AveragedPerceptron() {}
    string predict(const unordered_map<string, int>& features);

    unordered_map<string, unordered_map<string, double>> weights;
    unordered_set<string> classes;
    unordered_map<string, string> tagdict;

    void import_from_text();

    const vector<string> START = { "-START-", "-START2-" };
    const vector<string> END = { "-END-", "-END2-" };

    unordered_map<string, int> _get_features(int i, const string& word, const vector<string>& context, const string& prev, const string& prev2);

    string normalize(const string& word);

    vector<pair<string, string>> tag(const vector<string>& tokens);
};


int test_averaged_perceptron(void);
