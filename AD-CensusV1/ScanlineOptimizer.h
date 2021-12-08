#pragma once
#include<opencv2/opencv.hpp>	
#include<iostream>	

using namespace std;
using namespace cv;

class ScanlineOptimizer
{
public:
	ScanlineOptimizer();
	void Initialize(int row, int col, int dispRange, float* costVolume, int p1, int p2);
	void ScanLine(float* costVolume, float* Image);
	void	ScanLineLeftRight(float* costVolume, float* Image, bool isLeft);
	void ScanLineUpDown(float* costVolume, float* Image, bool isUp);
	float minvalueFunction(float a, float b, float c);
	void WTA(float* disp);
	~ScanlineOptimizer();

private:
	int _row;
	int _col;
	int _dispRange;
	float* _costVolume;
	float* _ProcessedVolume;
	float* leftVolume;
	float* rightVolume;
	float* upVolume;
	float* downVolume;
	//float* leftImage;
	//float* rightImage;
	int _p1;
	int _p2;
};

ScanlineOptimizer::ScanlineOptimizer()
{
}

void	ScanlineOptimizer::WTA(float* disp)
{
	int col = _col;
	int row = _row;
	int dispRange = _dispRange;
	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < col; j++)
		{
			auto cost = _ProcessedVolume[i * col * dispRange + j * dispRange + 0];
			float disparity = 0;
			for (int d = 1; d < dispRange; d++)
			{
				auto value = _ProcessedVolume[i * col * dispRange + j * dispRange + d];
				if (cost > value)
				{
					disparity = d;
					cost = value;
				}
			}
			auto& costDisp = disp[i * col + j];
			costDisp = disparity;
		}
	}
}

void ScanlineOptimizer::Initialize(int row, int col, int dispRange, float* costVolume, int p1, int p2)
{
	_row = row;
	_col = col;
	_dispRange = dispRange;
	_costVolume = costVolume;
	leftVolume = new float[col * row * dispRange]();
	rightVolume = new float[col * row * dispRange]();
	upVolume = new float[col * row * dispRange]();
	downVolume = new float[col * row * dispRange]();
	_ProcessedVolume = new float[col * row * dispRange]();
	_p1 = p1;
	_p2 = p2;
}

float ScanlineOptimizer::minvalueFunction(float a, float b, float c)
{
	if (a>b)
	{
		if (b > c)
			return c;
		else
		{
			return b;
		}
	}
	else
	{
		if (a > c)
			return c;
		else
		{
			return a;
		}
	}
}


void ScanlineOptimizer::ScanLine(float* costVolume, float* Image)
{
	int row = _row;
	int col = _col;
	int dispRange = _dispRange;
	ScanLineLeftRight(costVolume,Image, true); 
	ScanLineLeftRight(costVolume,Image, false);
	ScanLineUpDown(costVolume, Image, true);
	ScanLineUpDown(costVolume, Image, false);
	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < col; j++)
		{
			for (int d = 0; d < dispRange; d++)
			{
				auto left = leftVolume[i * col * dispRange + j * dispRange + d];
				auto right = rightVolume[i * col * dispRange + j * dispRange + d];
				auto up = upVolume[i * col * dispRange + j * dispRange + d];
				auto down = downVolume[i * col * dispRange + j * dispRange + d];
				auto& cost = _ProcessedVolume[i * col * dispRange + j * dispRange + d];
				cost = left + right + up + down;
			}
		}
	}
}

void ScanlineOptimizer::ScanLineLeftRight(float* costVolume, float* Image, bool isLeft)
{
	float p1 = _p1;
	float p2Init = _p2;
	int direction = isLeft ? 1 : -1; // 选择从左到右还是从右到左
	vector<float> costVec(_dispRange);  // 选择存储的容量大小
	for (int i = 0; i < _row; i++)
	{
		// 获取初始代价行
		auto costInitRow = isLeft ? (costVolume + i * _col * _dispRange) : (costVolume + i * _col * _dispRange + (_col - 1) * _dispRange);
		// 获取聚合代价行, 区分开来，一个是左一个是右
		auto costAggredRow = isLeft ? (leftVolume + i * _col * _dispRange) : (rightVolume + i * _col * _dispRange + (_col - 1) * _dispRange);
		// 获取灰度图像像素的位置，用于计算P2
		auto ImageGray = isLeft ? (Image + i * _col) : (Image + i * _col + (_col - 1));
		// 如果是正向的，那么从0开始，如果是反向的，那么从最后一列开始

		float gray = *ImageGray;
		float lastgray = *ImageGray;

		int x = isLeft ? 0 : _col - 1;
		// 存储数据，上一个路径上的代价
		vector<float> costLastPath(_dispRange + 2, 0xffff);
		// 第一列的代价作为初始的代价，将第一列的代价直接拷贝到聚合代价行的第一列上
		memcpy(costAggredRow, costInitRow, _dispRange * sizeof(float));
		// 记录上一个路径上的所有代价
		memcpy(&costLastPath[1], costAggredRow, _dispRange * sizeof(float));
		// 改变偏移的位置，计算下一个像素的代价
		costInitRow += direction * _dispRange;   //下一个代价的位置就是方向乘以视差范围，因为一个视差范围代表一个代价空间
		costAggredRow += direction * _dispRange;
		ImageGray += direction;
		x += direction;
		
		float minLastPath = 0xffff;  // 这个是用来最后的时候减去的一个值，防止总的代价值过大
		for (auto cost:costLastPath)
		{
			minLastPath = min(cost, minLastPath);  // 求上个路径的最小值
		}
		for (int j = 0; j < _col-1; j++)
		{
			float minCost = 0xffff;
			gray = *ImageGray;
			auto p2 = std::max(p1, p2Init / (abs(gray - lastgray) + 1)); 
			lastgray = gray;
			for (int d = 0; d < _dispRange; d++)
			{
				const float cost = costInitRow[d];
				const float l1 = costLastPath[d + 1];  // 因为开始的位置是从1开始的，所以所有的计算都要往右偏移一格
				const float l2 = costLastPath[d ] + p1;  // 这个相当于公式里面的d-1
				const float l3 = costLastPath[d + 2] + p1; // 这个相当于公式里面的d+1
				const float l4 = minLastPath + p2; 
				float cost_s = cost + min(min(l1, l2), min(l3, l4))-minLastPath;
				costAggredRow[d] = cost_s;
				minCost = min(minCost, cost_s);
			}
			minLastPath = minCost;  // 我感觉这一步的操作有点多余
			memcpy(&costLastPath[1], costAggredRow, _dispRange * sizeof(float));

			costInitRow += direction * _dispRange;
			costAggredRow += direction * _dispRange;
			ImageGray += direction;
		}
	}
}

void ScanlineOptimizer::ScanLineUpDown(float* costVolume, float* Image, bool isUp)
{
	float p1 = _p1;
	float p2Init = _p2;
	int direction = isUp ? 1 : -1; // 选择从左到右还是从右到左
	vector<float> costVec(_dispRange);  // 选择存储的容量大小
	for (int j = 0; j < _col; j++)
	{
		// 获取初始代价行
		auto costInitRow = isUp ? (costVolume + j * _dispRange) : (costVolume + j * _dispRange + ( _row - 1) * _col  * _dispRange);
		// 获取聚合代价行
		auto costAggredRow = isUp ? (upVolume + j * _dispRange) : (downVolume + j* _dispRange + (_row - 1)* _col * _dispRange);
		// 如果是正向的，那么从0开始，如果是反向的，那么从最后一列开始
		auto ImageGray = isUp ? (Image + j) : (Image + j + (_row - 1) * _col);

		auto gray = *ImageGray;
		auto grayLast = *ImageGray;
		int x = isUp ? 0 : _row - 1;
		// 存储数据，上一个路径上的代价 
		vector<float> costLastPath(_dispRange + 2, 0xffff);
		// 第一列的代价作为初始的代价，将第一列的代价直接拷贝到聚合代价行的第一列上
		memcpy(costAggredRow, costInitRow, _dispRange * sizeof(float));
		// 记录上一个路径上的所有代价
		memcpy(&costLastPath[1], costAggredRow, _dispRange * sizeof(float));
		// 改变便宜的位置，计算下一个像素的代价
		costInitRow += direction * _col * _dispRange;   //下一个代价的位置就是方向乘以视差范围，因为一个视差范围代表一个代价空间
		costAggredRow += direction* _col  * _dispRange;
		ImageGray += direction;
		//x += direction;

		float minLastPath = 0xffff;  // 这个是用来最后的时候减去的一个值，防止总的代价值过大
		for (auto cost : costLastPath)
		{
			minLastPath = min(cost, minLastPath);  // 求上个路径的最小值
		}
		for (int i = 0; i < _row-1; i++)
		{
			gray = *ImageGray;
			auto p2 = std::max(p1, p2Init / (abs(gray - grayLast) + 1));
			float minCost = 0xffff;
			for (int d = 0; d < _dispRange; d++)
			{
				const float cost = costInitRow[d];
				const float l1 = costLastPath[d + 1];
				const float l2 = costLastPath[d + 1] + p1;
				const float l3 = costLastPath[d + 2] + p1;
				const float l4 = minLastPath + p2;
				float cost_s = cost + min(min(l1, l2), min(l3, l4)) - minLastPath;
				costAggredRow[d] = cost_s;
				minCost = min(minCost, cost_s);
			}
			minLastPath = minCost;  // 我感觉这一步的操作有点多余
			memcpy(&costLastPath[1], costAggredRow, _dispRange * sizeof(float));

			costInitRow += direction* _col * _dispRange;
			costAggredRow += direction* _col * _dispRange;
			ImageGray += direction;
		}
	}
}

ScanlineOptimizer::~ScanlineOptimizer()
{
}