#include <iostream>
#include <fstream>
#include <vector>
#include <queue>
#include <unordered_set>
#include <chrono>
#include <cmath>
#include <cstring>
#include <algorithm>

// 引入 Path ORAM 相关的头文件
#include "OramInterface.h"
#include "OramReadPathEviction.h"
#include "ServerStorage.h"
#include "RandomForOram.h"
#include "Block.h"

using namespace std;

// ==========================================
// 1. 常量与定长数据结构定义
// ==========================================
const int DIMENSION = 128;        // 向量维度 (以 SIFT1M 为例)
const int MAX_DEGREE = 32;        // NSG 的最大出度限制 (R)
const int MAX_SEARCH_STEPS = 100; // 强制 ORAM 访问的固定步数 (防止侧信道泄露)
const int DUMMY_NODE_ID = 0;      // 用于填充假访问的冗余 ID

// 注意：此结构体的大小必须精确等于 Block.h 中的 BLOCK_SIZE * sizeof(int)
// 1 + 128 + 32 + 1 = 162 个 int/float，即 648 Bytes
struct FixedNSGNode
{
    int id;
    float vec[DIMENSION];
    int neighbors[MAX_DEGREE];
    int num_neighbors;
};

// ==========================================
// 2. 距离计算与数据序列化
// ==========================================
float calc_l2_sq(const float *a, const float *b, unsigned size)
{
    float res = 0;
    for (unsigned i = 0; i < size; i++)
    {
        float t = a[i] - b[i];
        res += t * t;
    }
    return res;
}

// 将 NSG 节点内存直接拷贝到 ORAM 需要的 int 数组中
void SerializeNode(const FixedNSGNode &node, int *out_data)
{
    memcpy(out_data, &node, sizeof(FixedNSGNode));
}

// 将 ORAM 返回的 int 数组解析回 NSG 节点
FixedNSGNode DeserializeNode(const int *in_data)
{
    FixedNSGNode node;
    memcpy(&node, in_data, sizeof(FixedNSGNode));
    return node;
}

// ==========================================
// 3. 数据集读取工具 (.fvecs / .ivecs)
// ==========================================
void load_fvecs(const char *filename, float *&data, unsigned &num, unsigned &dim)
{
    ifstream in(filename, ios::binary);
    if (!in.is_open())
    {
        cout << "Error opening " << filename << endl;
        exit(1);
    }
    in.read((char *)&dim, 4);
    in.seekg(0, ios::end);
    ios::pos_type ss = in.tellg();
    size_t fsize = (size_t)ss;
    num = (unsigned)(fsize / (dim + 1) / 4);
    data = new float[num * dim];
    in.seekg(0, ios::beg);
    for (size_t i = 0; i < num; i++)
    {
        in.seekg(4, ios::cur);
        in.read((char *)(data + i * dim), dim * 4);
    }
    in.close();
}

void load_ivecs(const char *filename, int *&data, unsigned &num, unsigned &dim)
{
    ifstream in(filename, ios::binary);
    if (!in.is_open())
    {
        cout << "Error opening " << filename << endl;
        exit(1);
    }
    in.read((char *)&dim, 4);
    in.seekg(0, ios::end);
    ios::pos_type ss = in.tellg();
    size_t fsize = (size_t)ss;
    num = (unsigned)(fsize / (dim + 1) / 4);
    data = new int[num * dim];
    in.seekg(0, ios::beg);
    for (size_t i = 0; i < num; i++)
    {
        in.seekg(4, ios::cur);
        in.read((char *)(data + i * dim), dim * 4);
    }
    in.close();
}

// ==========================================
// 4. 数据无关的图检索算法 (Oblivious NSG Search)
// ==========================================
struct Candidate
{
    int id;
    float distance;
    bool operator<(const Candidate &other) const { return distance < other.distance; }
    bool operator>(const Candidate &other) const { return distance > other.distance; }
};

vector<int> ObliviousSearch(
    const float *query,
    int ep_id,
    OramInterface *oram,
    int L,
    int K)
{
    priority_queue<Candidate, vector<Candidate>, less<Candidate>> top_results;
    priority_queue<Candidate, vector<Candidate>, greater<Candidate>> search_queue;
    unordered_set<int> visited;

    // 初始化一个全为 0 的数组，用于 READ 请求时的占位 (dummy newdata)
    int empty_data[Block::BLOCK_SIZE] = {0};

    bool is_converged = false;
    int next_target_id = ep_id;

    for (int step = 0; step < MAX_SEARCH_STEPS; step++)
    {

        // 1. 确定本次 ORAM 访问的目标：真访问 还是 假访问
        int access_id = (!is_converged && next_target_id != -1) ? next_target_id : DUMMY_NODE_ID;

        // 2. 发起 ORAM 读操作 (调用你在 OramInterface.h 中的 API)
        int *res_data = oram->access(OramInterface::READ, access_id, empty_data);

        if (!is_converged && access_id != DUMMY_NODE_ID)
        {
            visited.insert(access_id);

            // 解析返回的数据
            FixedNSGNode node = DeserializeNode(res_data);

            // 计算当前节点与 Query 的距离
            float dist = calc_l2_sq(query, node.vec, DIMENSION);

            // 更新结果池
            if (top_results.size() < L)
            {
                top_results.push({access_id, dist});
            }
            else if (dist < top_results.top().distance)
            {
                top_results.pop();
                top_results.push({access_id, dist});
            }

            // 将真实存在的邻居加入待访问队列
            for (int i = 0; i < MAX_DEGREE; i++)
            {
                int neighbor_id = node.neighbors[i];
                if (neighbor_id != -1 && visited.find(neighbor_id) == visited.end())
                {
                    search_queue.push({neighbor_id, 0.0f});
                    visited.insert(neighbor_id);
                }
            }
        }

        // 3. 决定下一步的搜索目标
        next_target_id = -1;
        while (!search_queue.empty())
        {
            int cand = search_queue.top().id;
            search_queue.pop();
            if (visited.find(cand) == visited.end())
            {
                next_target_id = cand;
                break;
            }
        }

        // 4. 判断收敛，若收敛则进入假访问模式
        if (next_target_id == -1)
        {
            is_converged = true;
        }
    }

    // 提取 Top K
    vector<int> res;
    while (!top_results.empty())
    {
        res.push_back(top_results.top().id);
        top_results.pop();
    }
    reverse(res.begin(), res.end());
    if (res.size() > K)
        res.resize(K);
    return res;
}

// ==========================================
// 5. Main 主控函数
// ==========================================
int main(int argc, char **argv)
{
    if (argc < 4)
    {
        cout << "Usage: ./nsg_oram base.fvecs query.fvecs gt.ivecs nsg_graph.nsg" << endl;
        return 0;
    }

    // 1. 加载数据
    float *base_data = nullptr, *query_data = nullptr;
    int *gt_data = nullptr;
    unsigned nb, dim, nq, dim2, ng, dim3;

    cout << "Loading datasets..." << endl;
    load_fvecs(argv[1], base_data, nb, dim);
    load_fvecs(argv[2], query_data, nq, dim2);
    load_ivecs(argv[3], gt_data, ng, dim3);

    if (dim != DIMENSION)
    {
        cout << "Error: Dimension mismatch! Code defined: " << DIMENSION << ", Data: " << dim << endl;
        return 1;
    }

    // 2. 初始化 ORAM (适配 OramReadPathEviction.h 中的构造函数)
    cout << "Initializing Path ORAM with " << nb << " blocks..." << endl;
    int bucket_size = 4; // Path ORAM 默认参数 Z=4
    RandForOramInterface *rand_gen = new RandomForOram();
    UntrustedStorageInterface *server = new ServerStorage(); // 注意：根据你 ServerStorage 的实际构造函数传参 (如果需要的话)

    OramInterface *oram = new OramReadPathEviction(server, rand_gen, bucket_size, nb);

    // 3. 将 NSG 图转换为定长 int 数组并写入 ORAM
    int temp_data[Block::BLOCK_SIZE];
    for (int i = 0; i < nb; i++)
    {
        FixedNSGNode node;
        node.id = i;
        memcpy(node.vec, base_data + i * dim, dim * sizeof(float));

        // TODO: 这里应从读取的 .nsg 文件中填写真实的图拓扑
        // 这里模拟填充全部为空 (-1)
        fill(node.neighbors, node.neighbors + MAX_DEGREE, -1);
        node.num_neighbors = 0;

        SerializeNode(node, temp_data);

        // 写入 ORAM (WRITE 操作)
        oram->access(OramInterface::WRITE, i, temp_data);
    }
    cout << "ORAM Initialization complete." << endl;

    // 4. 跑测试与评估 Recall
    int L = 40;
    int K = 10;
    int entry_point = 0; // 从建图结果中获取的入口节点 ID

    int correct = 0;
    cout << "Starting Oblivious Search for " << nq << " queries..." << endl;

    auto start_time = chrono::high_resolution_clock::now();

    for (unsigned i = 0; i < nq; i++)
    {
        vector<int> res = ObliviousSearch(query_data + i * dim, entry_point, oram, L, K);

        unordered_set<int> gt_set(gt_data + i * dim3, gt_data + i * dim3 + K);
        for (int r : res)
        {
            if (gt_set.find(r) != gt_set.end())
            {
                correct++;
            }
        }

        if ((i + 1) % 50 == 0)
            cout << "Processed " << (i + 1) << " queries..." << endl;
    }

    auto end_time = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end_time - start_time);

    // 5. 输出指标
    float recall = (float)correct / (nq * K);
    float qps = nq / (duration.count() / 1000.0);
    float latency = (float)duration.count() / nq;

    cout << "=============================" << endl;
    cout << "Recall@" << K << " : " << recall * 100 << " %" << endl;
    cout << "Avg Query Latency: " << latency << " ms" << endl;
    cout << "QPS: " << qps << " queries/sec" << endl;
    cout << "=============================" << endl;

    // 内存释放
    delete[] base_data;
    delete[] query_data;
    delete[] gt_data;
    delete oram;
    delete server;
    delete rand_gen;

    return 0;
}