#pragma once

#include <opencv2/opencv.hpp>	
#include <iostream>

using namespace std;
using namespace cv;

class CrossArmAggregation
{
public:
	CrossArmAggregation();
	void Initialize(int row, int col, float* leftImage, float* rightImage, int tao, int dispRange);
	//float* GetPtr();
	void ComputeLeftArmLength(const Mat& Image);
	void ComputeRightArmLength(const Mat& Image);
	void ComputeTopArmLength(const Mat& Image);
	void ComputeButtonArmLength(const Mat& Image);
	void Aggregation(float* dispVolume, float* aggregatedCostVolume);
	void AggregationVertical(float* dispVolume, float* aggregatedCostVolume);
	void WTA(float* AggredCostVolume, float* disp);
	~CrossArmAggregation();

private:
	int _col;
	int _row;
	float* _costVolum;
	float* _leftImage;
	float* _rightImage;
	int* leftLength;
	int* rightLength;
	int* topLength;
	int* buttonLenght;
	int _tao;
	int _dispRange;
};