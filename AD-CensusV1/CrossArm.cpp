#include "CrossArm.h"

CrossArmAggregation::CrossArmAggregation() :_col(0), _row(0), _costVolum(nullptr), _leftImage(nullptr), _rightImage(nullptr), leftLength(nullptr),
rightLength(nullptr), topLength(nullptr),buttonLenght(nullptr), _tao(0), _dispRange(0){}

void CrossArmAggregation::Initialize(int row, int col, float *leftImage, float* rightImage, int tao, int dispRange)
{
	_col = col;
	_row = row;
	_leftImage = leftImage;
	_rightImage = rightImage;
	_tao = tao;
	_dispRange = dispRange;
	leftLength = new int[col * row]();
	rightLength = new int[col * row]();
	topLength = new int[col * row]();
	buttonLenght = new int[col * row]();
}


//float* CrossArmAggregation::GetPtr()
//{
//	if (AggredCostVolume != nullptr)
//	{
//		return &AggredCostVolume[0];
//	}
//	else
//	{
//		return nullptr;
//	}
//}

void CrossArmAggregation::WTA(float* AggredCostVolume, float* disp)
{
	int col = _col;
	int row = _row;
	int dispRange = _dispRange;
	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < col; j++)
		{
			auto cost = AggredCostVolume[i * col * dispRange + j * dispRange + 0];
			float disparity = 0;
			for (int d = 1; d < dispRange; d++)
			{
				auto value = AggredCostVolume[i * col * dispRange + j * dispRange + d];
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

// 这个聚合方式是从上到下，然后从做到右
void CrossArmAggregation::AggregationVertical(float* dispVolume, float* aggregatedCostVolume)
{
	int col = _col;
	int row = _row;
	float* dispLayer = new float[col * row]();
	for (int d = 0; d < _dispRange; ++d)
	{
		// 将某一层的视差值取出来放在dispLayer中
		for (int i = 0; i < row; i++)
		{
			for (int j = 0; j < col; j++)
			{
				auto disp = dispVolume[i * col * _dispRange + j * _dispRange + d];
				dispLayer[i * col + j] = disp;
			}
		}
		// 聚合这一层的dispLayer
		for (int i = 0; i < row; i++)  
		{
			for (int j = 0; j < col; j++)
			{
				auto L = leftLength[i * col + j];
				auto R = rightLength[i * col + j];
				auto up = topLength[i * col + j];
				auto down = buttonLenght[i * col + j];
				float value = 0;
				int number = 0;
				// 这里差一个等号真的就天差地别 
				for (int left = -L; left <=R; left++) // 从上到下
				{
					for (int  top= -up; top <=down; top++)
					{
						value = value + dispLayer[(i + top) * col + j + left];
						number++;
					}
				}
				value = value / number;
				auto& cost = aggregatedCostVolume[i * col * _dispRange + j * _dispRange + d];
				cost = value;
			}
		}
	}
}

void CrossArmAggregation::Aggregation(float* dispVolume, float* aggregatedCostVolume)
{
	int col = _col;
	int row = _row;
	float* dispLayer = new float[col * row]();
	for (int d = 0; d < _dispRange; ++d)
	{
		// 将某一层的视差值取出来放在dispLayer中
		for (int i = 0; i < row; i++)
		{
			for (int j = 0; j < col; j++)
			{
				auto disp = dispVolume[i * col * _dispRange + j * _dispRange + d];
				dispLayer[i * col + j] = disp;
			}
		}
		// 聚合这一层的dispLayer
		for (int i = 0; i < row; i++)  // 丢掉一行
		{
			for (int j = 0; j < col; j++)
			{
				auto L = leftLength[i * col  + j];
				auto R = rightLength[i * col  + j];
				auto up = topLength[i * col + j];
				auto down = buttonLenght[i * col  + j];
				float value = 0;
				int number = 0;
				for (int top = -up; top < down; top++) // 从上到下
				{
					for (int left = -L; left < R; left++)
					{
						value = value + dispLayer[(i + top) * col + j + left];
						number++;
					}
				}
				value = value / number;
				auto& cost = aggregatedCostVolume[i * col * _dispRange + j * _dispRange + d];
				cost = value;
			}
		}
	}
}

void CrossArmAggregation::ComputeLeftArmLength(const Mat& Image)
{
	const int row = _row;
	const int col = _col;
	int savedNumber = 0;
	//#pragma omp parallel for  
	if (Image.channels() == 3)
	{
		for (int i = 0; i < row; ++i)
		{
			for (int j = 0; j < col; ++j)
			{
				int offset = 0;
				//vector<uchar> diffValue(3);
				auto x = Image.at<Vec3b>(i, j)[0];
				auto y = Image.at<Vec3b>(i, j)[1];
				auto z = Image.at<Vec3b>(i, j)[2];
				while (j - offset >= 0)
				{
					savedNumber = offset;
					offset++;
					if (offset > 17)
					{
						_tao = 6;
						if (offset > 34)
							break;
					}
					if (j - offset >= 0)
					{
						//savedNumber = offset;
						auto x_ = Image.at<Vec3b>(i, j - offset)[0];
						auto y_ = Image.at<Vec3b>(i, j - offset)[1];
						auto z_ = Image.at<Vec3b>(i, j - offset)[2];
						auto xValue = std::abs(x - x_);
						auto yValue = std::abs(y - y_);
						auto zValue = std::abs(z - z_);
						auto maxValue = max(max(xValue, yValue), zValue);
						if (maxValue > _tao)
						{
							//offset = 1;
							if ((j - 1 >= 1) && (savedNumber < 1))
							{
								savedNumber = 1;
								break;
							}
							else
							{
								break;
							}
							//break;
						}
					}
					else
					{
						//savedNumber = 1;
						break;
					}
				}
				leftLength[i * col + j] = savedNumber;
				savedNumber = 0;
			}
		}
	}
	else
	{
		for (int i = 0; i < row; ++i)
		{
			for (int j = 0; j < col; ++j)
			{
				int offset = 0;
				//vector<uchar> diffValue(3);
				auto x = Image.at<uchar>(i, j);
				while (j - offset >= 0)
				{
					savedNumber = offset;
					offset++;
					if (offset > 17)
					{
						_tao = 6;
						if (offset > 34)
							break;
					}
					if (j - offset >= 0)
					{
						//savedNumber = offset;
						auto x_ = Image.at<uchar>(i, j - offset);
						auto maxValue = std::abs(x - x_);
						if (maxValue > _tao)
						{
							//offset = 0;
							if ((j - 1 >= 1) && (savedNumber < 1))
							{
								savedNumber = 1;
								break;
							}
							else
							{
								break;
							}
							//break;
						}
					}
					else
					{
						//savedNumber = 1;
						break;
					}
				}
				leftLength[i * col + j] = savedNumber;
				savedNumber = 0;
			}
		}
	}
}

void CrossArmAggregation::ComputeRightArmLength(const Mat& Image)
{
	const int row = _row;
	const int col = _row;
	int savedNumber = 0;
	//#pragma omp parallel for  
	if (Image.channels() == 3)
	{
		for (int i = 0; i < row; ++i)
		{
			for (int j = 0; j < col; ++j)
			{
				int offset = 0;
				//vector<uchar> diffValue(3);
				auto x = Image.at<Vec3b>(i, j)[0];
				auto y = Image.at<Vec3b>(i, j)[1];
				auto z = Image.at<Vec3b>(i, j)[2];
				while (j + offset < col)
				{
					savedNumber = offset;
					offset++;
					if (offset > 17)
					{
						_tao = 6;
						if (offset > 34)
							break;
					}
					if (j + offset < col)
					{
						//savedNumber = offset;
						auto x_ = Image.at<Vec3b>(i, j + offset)[0];
						auto y_ = Image.at<Vec3b>(i, j + offset)[1];
						auto z_ = Image.at<Vec3b>(i, j + offset)[2];
						auto xValue = std::abs(x - x_);
						auto yValue = std::abs(y - y_);
						auto zValue = std::abs(z - z_);
						auto maxValue = max(max(xValue, yValue), zValue);
						if (maxValue > _tao)
						{
							//offset = 1;
							if ((j + 1 < col - 1) && (savedNumber < 1))
							{
								savedNumber = 1;
								break;
							}
							else
							{
								break;
							}
							//break;
						}
					}
					else
					{
						//savedNumber = 1;
						break;
					}
				}
				rightLength[i * col + j] = savedNumber;
				savedNumber = 0;
			}
		}
	}
	else
	{
		for (int i = 0; i < row; ++i)
		{
			for (int j = 0; j < col; ++j)
			{
				int offset = 0;
				auto x = Image.at<uchar>(i, j);
				while (j + offset < col)
				{
					savedNumber = offset;
					offset++;
					if (offset > 17)
					{
						_tao = 6;
						if (offset > 34)
							break;
					}
					if (j + offset < col)
					{
						auto x_ = Image.at<uchar>(i, j + offset);
						auto maxValue = std::abs(x - x_);
						if (maxValue > _tao)
						{
							//offset = 1;
							if ((j + 1 < col - 1) && (savedNumber < 1))
							{
								savedNumber = 1;
								break;
							}
							else
							{
								break;
							}
							//break;
						}
					}
					else
					{
						//savedNumber = 1;
						break;
					}
				}
				rightLength[i * col + j] = savedNumber;
				savedNumber = 0;
			}
		}
	}
}

void CrossArmAggregation::ComputeTopArmLength(const Mat& Image)
{
	const int row = _row;
	const int col = _col;
	int savedNumber = 0;
	//#pragma omp parallel for  
	if(Image.channels() == 3)
	{
		for (int i = 0; i < row; ++i)
		{
			for (int j = 0; j < col; ++j)
			{
				int offset = 0;
				//vector<uchar> diffValue(3);
				auto x = Image.at<Vec3b>(i, j)[0];
				auto y = Image.at<Vec3b>(i, j)[1];
				auto z = Image.at<Vec3b>(i, j)[2];
				while (i - offset >= 0)
				{
					savedNumber = offset;
					offset++;
					if (offset > 17)
					{
						_tao = 6;
						if (offset > 34)
							break;
					}
					if (i - offset >= 0)
					{
						//savedNumber = offset;
						auto x_ = Image.at<Vec3b>(i - offset, j)[0];
						auto y_ = Image.at<Vec3b>(i - offset, j)[1];
						auto z_ = Image.at<Vec3b>(i - offset, j)[2];
						auto xValue = std::abs(x - x_);
						auto yValue = std::abs(y - y_);
						auto zValue = std::abs(z - z_);
						auto maxValue = max(max(xValue, yValue), zValue);
						if (maxValue > _tao)
						{
							//offset = 0;
							if ((i - 1 >= 1) && (savedNumber < 1))
							{
								savedNumber = 1;
								break;
							}
							else
							{
								break;
							}
							//break;
						}
					}
					else
					{
						//savedNumber = 1;
						break;
					}
				}
				topLength[i * col + j] = savedNumber;
				savedNumber = 0;
			}
		}
	}
	else
	{
		for (int i = 0; i < row; ++i)
		{
			for (int j = 0; j < col; ++j)
			{
				int offset = 0;
				auto x = Image.at<uchar>(i, j);
				while (i - offset >= 0)
				{
					savedNumber = offset;
					offset++;
					if (offset > 17)
					{
						_tao = 6;
						if (offset > 34)
							break;
					}
					if (i - offset >= 0)
					{
						auto x_ = Image.at<uchar>(i - offset, j);
						auto maxValue = std::abs(x - x_);
						if (maxValue > _tao)
						{
							//offset = 0;
							if ((i - 1 >= 1) && (savedNumber < 1))
							{
								savedNumber = 1;
								break;
							}
							else
							{
								break;
							}
							//break;
						}
					}
					else
					{
						//savedNumber = 1;
						break;
					}
				}
				topLength[i * col + j] = savedNumber;
				savedNumber = 0;
			}
		}
	}
}

void CrossArmAggregation::ComputeButtonArmLength(const Mat& Image)
{
	const int row = _row;
	const int col = _col;
	int savedNumber = 0;
	//#pragma omp parallel for  
	if (Image.channels() == 3)
	{
		for (int i = 0; i < row; ++i)
		{
			for (int j = 0; j < col; ++j)
			{
				int offset = 0;
				//vector<uchar> diffValue(3);
				auto x = Image.at<Vec3b>(i, j)[0];
				auto y = Image.at<Vec3b>(i, j)[1];
				auto z = Image.at<Vec3b>(i, j)[2];
				while (i + offset < row)
				{
					savedNumber = offset;
					offset++;
					if (offset > 17)
					{
						_tao = 6;
						if (offset > 34)
							break;
					}
					if (i + offset < row)
					{
						//savedNumber = offset;
						auto x_ = Image.at<Vec3b>(i + offset, j)[0];
						auto y_ = Image.at<Vec3b>(i + offset, j)[1];
						auto z_ = Image.at<Vec3b>(i + offset, j)[2];
						auto xValue = std::abs(x - x_);
						auto yValue = std::abs(y - y_);
						auto zValue = std::abs(z - z_);
						auto maxValue = max(max(xValue, yValue), zValue);
						if (maxValue > _tao)
						{
							//offset = 0;
							if ((i + 1 < row - 1) && (savedNumber < 1))
							{
								savedNumber = 1;
								break;
							}
							else
							{
								break;
							}
							//break;
						}
					}
					else
					{
						//savedNumber = 1;
						break;
					}
				}
				buttonLenght[i * col + j] = savedNumber;
				savedNumber = 0;
			}
		}
	}
	else
	{
		for (int i = 0; i < row; ++i)
		{
			for (int j = 0; j < col; ++j)
			{
				int offset = 0;
				//vector<uchar> diffValue(3);
				auto x = Image.at<uchar>(i, j);
				while (i + offset < row)
				{
					savedNumber = offset;
					offset++;
					if (offset > 17)
					{
						_tao = 6;
						if (offset > 34)
							break;
					}
					if (i + offset < row)
					{
						//savedNumber = offset;
						auto x_ = Image.at<uchar>(i + offset, j);
						auto maxValue = std::abs(x - x_);
						if (maxValue > _tao)
						{
							if ((i + 1 < row - 1) && (savedNumber < 1))
							{
								savedNumber = 1;
								break;
							}
							else
							{
								break;
							}
						}
					}
					else
					{
						break;
					}
				}
				buttonLenght[i * col + j] = savedNumber;
				savedNumber = 0;
			}
		}
	}
}

CrossArmAggregation::~CrossArmAggregation()
{
}