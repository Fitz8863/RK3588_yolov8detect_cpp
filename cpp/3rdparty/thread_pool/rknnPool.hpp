#ifndef RKNNPOOL_H
#define RKNNPOOL_H

#include "ThreadPool.hpp"
#include <vector>
#include <iostream>
#include <mutex>
#include <queue>
#include <memory>

// rknnModel模型类, inputType模型输入类型, outputType模型输出类型
template <typename rknnModel, typename inputType, typename outputType>
class rknnPool
{
private:
    int threadNum;
    std::string modelPath;
    long long int id;
    
    std::mutex idMtx, queueMtx;
    std::unique_ptr<ThreadPool> pool;
    std::queue<std::future<outputType>> futs;
    std::vector<std::shared_ptr<rknnModel>> models;

protected:
    int getModelId();

public:
    rknnPool(const std::string modelPath, int threadNum);
    int init();
    // 模型推理/Model inference
    int put(inputType inputData);
    // 获取推理结果/Get the results of your inference
    int get(outputType& outputData);
    ~rknnPool();
};

template <typename rknnModel, typename inputType, typename outputType>
rknnPool<rknnModel, inputType, outputType>::rknnPool(const std::string modelPath, int threadNum)
{
    this->modelPath = modelPath;
    this->threadNum = threadNum;
    this->id = 0;
}

template <typename rknnModel, typename inputType, typename outputType>
int rknnPool<rknnModel, inputType, outputType>::init()
{
    try
    {
        this->pool = std::make_unique<ThreadPool>(this->threadNum);
        for (int i = 0; i < this->threadNum; i++) {
            models.push_back(std::make_shared<rknnModel>(this->modelPath.c_str()));
            // std::cout<<"push 成功"<<std::endl;
        }
    }
    // 处理错误情况
    catch (const std::bad_alloc& e)
    {
        std::cout << "Out of memory: " << e.what() << std::endl;
        return -1;
    }
    
    // 初始化权重/Initialize the model
    for (int i = 0, ret = 0; i < this->threadNum; i++)
    {
        ret = models[i]->init_yolov8_model((models[0]->Get_app_ctx()), i!=0);   // init(models[0]->get_pctx(), i != 0);
        if (ret != 0){
            // std::cout<<"这里出问题了"<<std::endl;
            return ret;
        }
        // std::cout<<"线程池初始化出问题"<<std::endl;
    }
    return 0;
}

template <typename rknnModel, typename inputType, typename outputType>
int rknnPool<rknnModel, inputType, outputType>::getModelId()
{
    std::lock_guard<std::mutex> lock(idMtx);
    int modelId = id % threadNum;
    id++;
    return modelId;
}

template <typename rknnModel, typename inputType, typename outputType>
int rknnPool<rknnModel, inputType, outputType>::put(inputType inputData)
{
    std::lock_guard<std::mutex> lock(queueMtx);
    futs.push(pool->enqueue(&rknnModel::inference_yolov8_model, models[this->getModelId()], inputData));
    return 0;
}

template <typename rknnModel, typename inputType, typename outputType>
int rknnPool<rknnModel, inputType, outputType>::get(outputType& outputData)
{
    std::lock_guard<std::mutex> lock(queueMtx);
    if (futs.empty() == true)
        return 1;
    outputData = futs.front().get();
    futs.pop();
    return 0;
}

template <typename rknnModel, typename inputType, typename outputType>
rknnPool<rknnModel, inputType, outputType>::~rknnPool()
{
    while (!futs.empty())
    {
        outputType temp = futs.front().get();
        futs.pop();
    }
}
#endif
