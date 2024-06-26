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
    string predict(const unordered_map<string, int>& features) {
        unordered_map<string, double> scores;
        for (const auto& feat : features) {
            if (weights.find(feat.first) == weights.end() || feat.second == 0) continue;
            const unordered_map<string, double>& label_weights = weights[feat.first];
            for (const auto& lw : label_weights) {
                scores[lw.first] += feat.second * lw.second;
            }
        }
        string best_label = *max_element(classes.begin(), classes.end(), [&](const string& a, const string& b) {
            return scores[a] < scores[b] || (scores[a] == scores[b] && a > b);
        });
        return best_label;
    }
    unordered_map<string, unordered_map<string, double>> weights;
    unordered_set<string> classes;
};

AveragedPerceptron model;

tuple<unordered_map<string, unordered_map<string, double>>, unordered_map<string, string>, unordered_set<string>> import_from_text() {
    unordered_map<string, unordered_map<string, double>> weights;
    unordered_map<string, string> tagdict;
    unordered_set<string> classes;

    ifstream weights_file("averaged_perceptron_tagger_weights.txt");
    string line, feat, label;
    double weight;
    while (getline(weights_file, line)) {
        istringstream iss(line);
        getline(iss, feat, ' ');
        getline(iss, label, ' ');
        iss >> weight;
        weights[feat][label] = weight;
    }
    weights_file.close();

    ifstream tagdict_file("averaged_perceptron_tagger_tagdict.txt");
    while (getline(tagdict_file, line)) {
        istringstream iss(line);
        string tag, v;
        getline(iss, tag, ' ');
        getline(iss, v, ' ');
        tagdict[tag] = v;
    }
    tagdict_file.close();

    ifstream classes_file("averaged_perceptron_tagger_classes.txt");
    while (getline(classes_file, line)) {
        classes.insert(line);
    }
    classes_file.close();
    return { weights, tagdict, classes };
}

const vector<string> START = { "-START-", "-START2-" };
const vector<string> END = { "-END-", "-END2-" };

unordered_map<string, int> _get_features(int i, const string& word, const vector<string>& context, const string& prev, const string& prev2) {
    unordered_map<string, int> features;

    auto add = [&features](const string& name, const vector<string>& args = {}) {
        string key = name;
        for (const auto& arg : args) key += " " + arg;
        features[key]++;
    };

    i += START.size();
    add("bias");
    add("i suffix", { word.substr(word.size() - 3) });
    add("i pref1", { word.empty() ? "" : string(1, word[0]) });
    add("i-1 tag", { prev });
    add("i-2 tag", { prev2 });
    add("i tag+i-2 tag", { prev, prev2 });
    add("i word", { context[i] });
    add("i-1 tag+i word", { prev, context[i] });
    add("i-1 word", { context[i - 1] });
    add("i-1 suffix", { context[i - 1].substr(context[i - 1].size() - 3) });
    add("i-2 word", { context[i - 2] });
    add("i+1 word", { context[i + 1] });
    add("i+1 suffix", { context[i + 1].substr(context[i + 1].size() - 3) });
    add("i+2 word", { context[i + 2] });

    return features;
}

string normalize(const string& word) {
    if (word.find('-') != string::npos && word[0] != '-') {
        return "!HYPHEN";
    }
    if (all_of(word.begin(), word.end(), ::isdigit) && word.size() == 4) {
        return "!YEAR";
    }
    if (!word.empty() && ::isdigit(word[0])) {
        return "!DIGITS";
    }
    string lower_word;
    transform(word.begin(), word.end(), back_inserter(lower_word), ::tolower);
    return lower_word;
}

vector<pair<string, string>> tag(const vector<string>& tokens, const unordered_map<string, string>& tagdict) {
    string prev = START[0], prev2 = START[1];
    vector<pair<string, string>> output;

    vector<string> context = START;
    for (const auto& w : tokens) {
        context.push_back(normalize(w));
    }
    context.insert(context.end(), END.begin(), END.end());

    for (int i = 0; i < tokens.size(); ++i) {
        string word = tokens[i];
        string tag;
        if (tagdict.find(word) != tagdict.end()) {
            tag = tagdict.at(word);
        } else {
            auto features = _get_features(i, word, context, prev, prev2);
            tag = model.predict(features);
        }
        output.push_back({ word, tag });

        prev2 = prev;
        prev = tag;
    }

    return output;
}

int test() {
    bool UNIT_TEST = true;

    if (UNIT_TEST) {
        auto [weights, tagdict, classes] = import_from_text();
        model.weights = weights;
        model.classes = classes;

        vector<string> words = { "i'm", "an", "activationist", "." };
        auto output = tag(words, tagdict);

        for (const auto& pair : output) {
            cout << "(" << pair.first << ", " << pair.second << "), ";
        }
        cout << endl;
    }

    return 0;
}
