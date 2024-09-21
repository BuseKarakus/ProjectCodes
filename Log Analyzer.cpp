#include <iostream>
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>
#include <chrono>
#include <algorithm>

class HashTable {
private:
    static const size_t TABLE_SIZE = 1000;
    std::vector<std::unordered_map<std::string, int>> table;

    size_t customHash(const std::string& key) const {

        size_t hash = 5381;
        for (char c : key) {
            hash = (hash << 5) + hash + c;
        }
        return hash % TABLE_SIZE;
    }

public:
    HashTable() : table(TABLE_SIZE) {

        for (auto& bucket : table) {
            bucket.reserve(10);
        }
    }

    void insert(const std::string& key) {
        size_t index = customHash(key);
        auto& bucket = table[index];
        auto it = bucket.find(key);
        if (it != bucket.end()) {
            it->second++;
        } else {
            bucket.emplace(key, 1);
        }
    }

    std::vector<std::pair<int, std::string>> getTopPages() const {
        std::vector<std::pair<int, std::string>> result;
        result.reserve(TABLE_SIZE * 10);

        for (const auto& bucket : table) {
            for (const auto& entry : bucket) {
                result.emplace_back(entry.second, entry.first);
            }
        }

        std::partial_sort(result.begin(), result.begin() + std::min(result.size(), size_t(10)), result.end(), std::greater<>());

        return result;
    }
};

void processLogFileWithCustomHashTable(const std::string& filename, std::chrono::duration<double>& customHashTableDuration) {
    HashTable customHashTable;
    auto startCustomHashTable = std::chrono::high_resolution_clock::now();

    std::ifstream logFile(filename);
    if (!logFile.is_open()) {
        std::cerr << "Error opening log file: " << filename << "\n";
        return;
    }

    std::string line;
    while (std::getline(logFile, line)) {
        size_t getPos = line.find("GET");
        if (getPos != std::string::npos) {
            size_t startPos = getPos + 4;
            size_t endPos = line.find(' ', startPos);
            std::string page = line.substr(startPos, endPos - startPos);
            customHashTable.insert(page);
        }
    }

    logFile.close();

    auto endCustomHashTable = std::chrono::high_resolution_clock::now();
    customHashTableDuration = endCustomHashTable - startCustomHashTable;

    auto topPages = customHashTable.getTopPages();
    int count = 1;
    for (const auto& entry : topPages) {
        std::cout << count << ": " << entry.second << " #" << entry.first << "\n";
        count++;
        if (count > 10) {
            break;
        }
    }
}

void processLogFileWithStdUnorderedMap(const std::string& filename, std::chrono::duration<double>& stdUnorderedMapDuration) {
    std::unordered_map<std::string, int> pageCounts;
    auto startStdUnorderedMap = std::chrono::high_resolution_clock::now();
    std::ifstream logFile(filename);
    if (!logFile.is_open()) {
        std::cerr << "Error opening log file: " << filename << "\n";
        return;
    }
    std::string line;
    while (std::getline(logFile, line)) {
        size_t getPos = line.find("GET");
        if (getPos != std::string::npos) {
            size_t startPos = getPos + 4;
            size_t endPos = line.find(' ', startPos);
            std::string page = line.substr(startPos, endPos - startPos);
            pageCounts[page]++;
        }
    }
    logFile.close();
    auto endStdUnorderedMap = std::chrono::high_resolution_clock::now();
    stdUnorderedMapDuration = endStdUnorderedMap - startStdUnorderedMap;

    std::vector<std::pair<int, std::string>> topPages;
    for (const auto& entry : pageCounts) {
        topPages.emplace_back(entry.second, entry.first);
    }
    std::sort(topPages.begin(), topPages.end(), std::greater<>());

    int count = 1;
    for (const auto& entry : topPages) {
        std::cout << count << ": " << entry.second << " #" << entry.first << "\n";
        count++;
        if (count > 10) {
            break;
        }
    }
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <access_log.txt>\n";
        return 1;
    }
    std::cout<<"results for Unordered Map :" ;std::cout<<std::endl;
    std::chrono::duration<double> totalDuration;
    auto startStdUnorderedMap = std::chrono::high_resolution_clock::now();
    processLogFileWithStdUnorderedMap(argv[1], totalDuration);
    auto endStdUnorderedMap = std::chrono::high_resolution_clock::now();
    auto durationStdUnorderedMap = std::chrono::duration_cast<std::chrono::duration<double>>(endStdUnorderedMap - startStdUnorderedMap);
    std::cout << "Total Elapsed Time for Unordered Map: " << durationStdUnorderedMap.count() << " seconds\n";
    std:: cout << std::endl ;

    totalDuration += durationStdUnorderedMap;


    std::cout<<"results for CustomHashTable: "; std::cout<<std::endl;
    std::chrono::duration<double> customHashTableDuration;
    auto startCustomHashTable = std::chrono::high_resolution_clock::now();
    processLogFileWithCustomHashTable(argv[1], customHashTableDuration);
    auto endCustomHashTable = std::chrono::high_resolution_clock::now();
    auto durationCustomHashTable = std::chrono::duration_cast<std::chrono::duration<double>>(endCustomHashTable - startCustomHashTable);
    std::cout << "Total Elapsed Time for CustomHashTable: " << durationCustomHashTable.count() << " seconds\n";

    totalDuration += durationCustomHashTable;



    std::cout << "\nTotal Elapsed Time for the Entire Process: " << totalDuration.count() << " seconds\n";

    return 0;
}