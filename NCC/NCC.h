#pragma once
#include<iostream>	
#include<opencv2/opencv.hpp>	
using namespace cv;
using namespace std;

#define invalid 0xffui8

void winMean(const Mat& win_, uchar& mean)
{
	auto temple = cv::mean(win_);
	mean = temple.val[0];
}

void ComputeCost(const Mat& win_L, const Mat& win_R, double& cost)
{
    auto win_row = win_L.rows;
    auto win_col = win_L.cols;
    double left_mean = 0, sum_left = 0;
    double right_mean = 0, sum_right = 0;
    double left_std = 0, right_std = 0;
    double numerator = 0;
    int n = win_row * win_col;
    double fenmu = 0;
    for (auto i = 0; i < win_row; i++)
    {
        for (auto j = 0; j < win_col; j++)
        {
            left_mean += win_L.at<uchar>(j,i);
            right_mean += win_R.at<uchar>(j,i);
        }
    }
    left_mean /= n;
    right_mean /= n;

    for (auto i = 0; i < win_row; i++)
    {
        for (auto j = 0; j < win_col; j++)
        {
            left_std += pow(win_L.at<uchar>(i, j) - left_mean, 2);
            right_std += pow(win_R.at<uchar>(i, j) - right_mean, 2);
            numerator += (win_L.at<uchar>(i, j) - left_mean) * (win_R.at<uchar>(i, j) - right_mean);
        }
    }

    cost = numerator / (sqrt(left_std) * sqrt(right_std));
    
    
}



int WinTakeAll(vector<double> disp)
{
	int best_disp= 0;
	float minDisp = disp[0];
	for (auto i = 1; i < disp.size(); ++i)
	{
		if (minDisp < disp[i])
		{
			best_disp = i;
			minDisp = disp[i];
		}
	}
	
	return best_disp;
}

void NCC_algorithem(const Mat& leftImage, const Mat& rigthImage, const int width, const int height, Mat& disp,
	int winSize, int dispRange)
{
	for (int i = winSize; i < height - winSize; ++i)
	{
		int *ptr_disp = disp.ptr<int>(i);
		for (int j = winSize; j < width-winSize; ++j)
		{
			auto win_L = leftImage(Range(i - winSize, i + winSize + 1), Range(j - winSize, j + winSize+1));
			vector<double> cost(dispRange);
			for (auto d = 0; d < dispRange; ++d)
			{
				if ((j-winSize-d) >= 0)
				{
					auto win_R = rigthImage(Range(i - winSize, i + winSize + 1), Range(j - winSize - d, j + winSize - d + 1));
					ComputeCost(win_L, win_R, cost[d]);
				}
				else
				{
					cost[d] = invalid;
				}
			}
			*(ptr_disp + j) = WinTakeAll(cost);
		}
        cout << "process is :" << float (i)/float(height-2*winSize)*100<< "%" << endl;
	}
}

Mat bgr_to_grey(const Mat& bgr)
{
    int width = bgr.size().width;
    int height = bgr.size().height;
    Mat grey(height, width, 0);

    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            uchar r = 0.333 * bgr.at<Vec3b>(y, x)[2];
            uchar g = 0.333 * bgr.at<Vec3b>(y, x)[1];
            uchar b = 0.333 * bgr.at<Vec3b>(y, x)[0];
            grey.at<uchar>(y, x) = uchar(r + g + b);
        }
    }

    return grey;
}

Mat ncc(Mat in1, Mat in2, string type, bool add_constant = false)
{
    int width = in1.size().width;
    int height = in1.size().height;
    int max_offset = 79;
    int kernel_size = 5; // window size

    Mat left = in1;
    Mat right = in2;

    if (add_constant)
    {
        right += 10;
    }

    Mat depth(height, width, 0);
    vector< vector<double> > max_ncc; // store max NCC value

    for (int i = 0; i < height; ++i)
    {
        vector<double> tmp(width, -2);
        max_ncc.push_back(tmp);
    }

    for (int offset = 1; offset <= max_offset; offset++)
    {
        Mat tmp(height, width, 0);
        // shift image depend on type to save calculation time
        if (type == "left")
        {
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < offset; x++)
                {
                    tmp.at<uchar>(y, x) = right.at<uchar>(y, x);
                }

                for (int x = offset; x < width; x++)
                {
                    tmp.at<uchar>(y, x) = right.at<uchar>(y, x - offset);  // 不太懂为什么这里要这样子操作，好像就是复制了两次右图的一半
                }
            }
        }
        else if (type == "right")
        {
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width - offset; x++)
                {
                    tmp.at<uchar>(y, x) = left.at<uchar>(y, x + offset);
                }

                for (int x = width - offset; x < width; x++)
                {
                    tmp.at<uchar>(y, x) = left.at<uchar>(y, x);
                }
            }
        }
        else
        {
            Mat tmp(0, 0, 0);
            return tmp;
        }

        // calculate each pixel's NCC value
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                int start_x = max(0, x - kernel_size);
                int start_y = max(0, y - kernel_size);
                int end_x = min(width - 1, x + kernel_size);
                int end_y = min(height - 1, y + kernel_size);
                double n = (end_y - start_y) * (end_x - start_x);  //统计一个窗口内元素的个数
                double res_ncc = 0;

                if (type == "left")
                {
                    double left_mean = 0, right_mean = 0;
                    double left_std = 0, right_std = 0;
                    double numerator = 0;

                    for (int i = start_y; i <= end_y; i++)
                    {
                        for (int j = start_x; j <= end_x; j++)
                        {
                            left_mean += left.at<uchar>(i, j);
                            right_mean += tmp.at<uchar>(i, j);
                        }
                    }

                    left_mean /= n;  // 所有元素之和除以元素的个数，得到均值
                    right_mean /= n;

                    for (int i = start_y; i <= end_y; i++)
                    {
                        for (int j = start_x; j <= end_x; j++)
                        {
                            left_std += pow(left.at<uchar>(i, j) - left_mean, 2);  
                            right_std += pow(tmp.at<uchar>(i, j) - right_mean, 2);
                            numerator += (left.at<uchar>(i, j) - left_mean) * (tmp.at<uchar>(i, j) - right_mean);
                        }
                    }

                    numerator /= n;
                    left_std /= n;
                    right_std /= n;
                    res_ncc = numerator / (sqrt(left_std) * sqrt(right_std)) / n;
                }
                else
                {
                    double left_mean = 0, right_mean = 0;
                    double left_std = 0, right_std = 0;
                    double numerator = 0;

                    for (int i = start_y; i <= end_y; i++)
                    {
                        for (int j = start_x; j <= end_x; j++)
                        {
                            left_mean += tmp.at<uchar>(i, j);
                            right_mean += right.at<uchar>(i, j);
                        }
                    }

                    left_mean /= n;
                    right_mean /= n;

                    for (int i = start_y; i <= end_y; i++)
                    {
                        for (int j = start_x; j <= end_x; j++)
                        {
                            left_std += pow(tmp.at<uchar>(i, j) - left_mean, 2);
                            right_std += pow(right.at<uchar>(i, j) - right_mean, 2);
                            numerator += (tmp.at<uchar>(i, j) - left_mean) * (right.at<uchar>(i, j) - right_mean);
                        }
                    }

                    numerator /= n;
                    left_std /= n;
                    right_std /= n;
                    res_ncc = numerator / (sqrt(left_std) * sqrt(right_std)) / n;
                }

                // greater NCC value found
                if (res_ncc > max_ncc[y][x])
                {
                    max_ncc[y][x] = res_ncc;
                    // for better visualization
                    depth.at<uchar>(y, x) = (uchar)(offset * 3);
                }
            }
        }
    }

    return depth;
}