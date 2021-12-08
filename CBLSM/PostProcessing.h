#pragma once
#include <opencv2/opencv.hpp>	
#include <iostream>

using namespace std;
using namespace cv;
//constexpr auto Invalid_Float = std::numeric_limits<float>::infinity();


void LeftAndRightConsistency(float* leftDisp, float* rightDisp,float* lastDisp, int col, int row, float gate,
	vector<pair<int, int>>&occlusion, vector<pair<int, int>> &mismatch)
{
	 // 左右一致性检查
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            // 左影像视差值
			auto& lastvalue = lastDisp[i * col + j];
        	auto& disp = leftDisp[i * col + j];
			//if(disp == Invalid_Float){             // 这个地方要改计算视差的程序
			//	mismatches.emplace_back(i, j);
			//	continue;
			//}

            // 根据视差值找到右影像上对应的同名像素
        	const auto col_right = static_cast<int>(j - disp + 0.5);  // 为什么加0.5 整形  因为这里是强制类型转换
            
        	if(col_right >= 0 && col_right < col) 
			{
                // 右影像上同名像素的视差值
                const auto& disp_r = rightDisp[i * col + col_right];
                
        		// 判断两个视差值是否一致（差值在阈值内）
        		if (abs(disp - disp_r) >= gate) {
					// 区分遮挡区和误匹配区
					// 通过右影像视差算出在左影像的匹配像素，并获取视差disp_rl
					// if(disp_rl > disp) 
        			//		pixel in occlusions
					// else 
        			//		pixel in mismatches
					const int col_rl = static_cast<int>(col_right + disp_r + 0.5);
					if(col_rl > 0 && col_rl < col){
						const auto& disp_l = leftDisp[i*col + col_rl];
						if(disp_l > disp) {
							occlusion.emplace_back(i, j);
						}
						else {
							mismatch.emplace_back(i, j);
						}
					}
					else{
						mismatch.emplace_back(i, j);
					}

                    // 让视差值无效
					//disp = 0;
					lastvalue = 0;
                }
				else
				{
					lastvalue = leftDisp[i * col + j];
				}
            }
            else{
                // 通过视差值在右影像上找不到同名像素（超出影像范围）
                lastvalue = 0;
				mismatch.emplace_back(i, j);
            }
        }
    }
}

void LeftRightConsistency(int col, int row, int gate,float* leftDisp, float* rightDisp, vector<pair<int, int>>& occlusion, vector<pair<int, int>>& mismatch)
{
	const int width = col;
	const int height = row;

	const float& threshold = gate;

	// 遮挡区像素和误匹配区像素
	auto& occlusions = occlusion;
	auto& mismatches = mismatch;
	occlusions.clear();
	mismatches.clear();

	// ---左右一致性检查
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			// 左影像视差值
			auto& disp = leftDisp[i * width + j];
			if (disp == Invalid_Float) {
				mismatches.emplace_back(i, j);
				continue;
			}

			// 根据视差值找到右影像上对应的同名像素
			const auto col_right = static_cast<int>(j - disp + 0.5);  // 为什么加0.5 整形 

			if (col_right >= 0 && col_right < width) {
				// 右影像上同名像素的视差值
				const auto& disp_r = rightDisp[i * width + col_right];

				// 判断两个视差值是否一致（差值在阈值内）
				if (abs(disp - disp_r) > threshold) {
					// 区分遮挡区和误匹配区
					// 通过右影像视差算出在左影像的匹配像素，并获取视差disp_rl
					// if(disp_rl > disp) 
					//		pixel in occlusions
					// else 
					//		pixel in mismatches
					const int col_rl = static_cast<int>(col_right + disp_r + 0.5);
					if (col_rl > 0 && col_rl < width) {
						const auto& disp_l = leftDisp[i * width + col_rl];
						if (disp_l > disp) {
							occlusions.emplace_back(i, j);
						}
						else {
							mismatches.emplace_back(i, j);
						}
					}
					else {
						mismatches.emplace_back(i, j);
					}

					// 让视差值无效
					disp = Invalid_Float;
				}
			}
			else {
				// 通过视差值在右影像上找不到同名像素（超出影像范围）
				disp = Invalid_Float;
				mismatches.emplace_back(i, j);
			}
		}
	}
}

void TransformToShow(float* disp, Mat& Taget, int row, int col)
{
	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < col; j++)
		{
			const int dispValue = disp[i * col + j];
			if (dispValue == 0)
			{
				Taget.data[i * col + j] = 0;
			}
			else
			{
				Taget.at<uchar>(i, j) = static_cast<uchar>(dispValue);
			}
		}
	}
}

void FillTheHole(const int row, const int col, const int dispRange, float* dispLeft, vector<pair<int, int>>&occlusion, vector<pair<int,int>>& mismatch)
{
	const int width = row;
	const int height = col;

	std::vector<float> disp_collects;

	// 定义8个方向
	const float pi = 3.1415926f;
	float angle1[8] = { pi, 3 * pi / 4, pi / 2, pi / 4, 0, 7 * pi / 4, 3 * pi / 2, 5 * pi / 4 };
	float angle2[8] = { pi, 5 * pi / 4, 3 * pi / 2, 7 * pi / 4, 0, pi / 4, pi / 2, 3 * pi / 4 };
	float* angle = angle1;
	// 最大搜索行程，没有必要搜索过远的像素
	const int max_search_length = 1.0 * dispRange;

	float* disp_ptr = dispLeft;
	for (int k = 0; k < 3; k++) {
		// 第一次循环处理遮挡区，第二次循环处理误匹配区
		auto& trg_pixels = (k == 0) ? occlusion : mismatch;
		if (trg_pixels.empty()) {
			continue;
		}
		std::vector<float> fill_disps(trg_pixels.size());
		std::vector<std::pair<int, int>> inv_pixels;
		if (k == 2) {
			//  第三次循环处理前两次没有处理干净的像素
			for (int i = 0; i < height; i++) {
				for (int j = 0; j < width; j++) {
					if (disp_ptr[i * width + j] == 0xffff) {
						inv_pixels.emplace_back(i, j);
					}
				}
			}
			trg_pixels = inv_pixels;
		}

		// 遍历待处理像素
		for (auto n = 0u; n < trg_pixels.size(); n++) {
			auto& pix = trg_pixels[n];
			const int y = pix.first;
			const int x = pix.second;

			if (y == height / 2) {
				angle = angle2;
			}

			// 收集8个方向上遇到的首个有效视差值
			disp_collects.clear();
			for (int s = 0; s < 8; s++) {
				const float ang = angle[s];
				const float sina = float(sin(ang));
				const float cosa = float(cos(ang));
				for (int m = 1; m < max_search_length; m++) {
					const int yy = lround(y + m * sina);
					const int xx = lround(x + m * cosa);
					if (yy < 0 || yy >= height || xx < 0 || xx >= width) {
						break;
					}
				    auto& disp = *(disp_ptr + yy * width + xx);
					if (disp != 0xffff) {
						disp_collects.push_back(disp);
						break;
					}
				}
			}
			if (disp_collects.empty()) {
				continue;
			}

			std::sort(disp_collects.begin(), disp_collects.end());

			// 如果是遮挡区，则选择第二小的视差值
			// 如果是误匹配区，则选择中值
			if (k == 0) {
				if (disp_collects.size() > 1) {
					fill_disps[n] = disp_collects[1];
				}
				else {
					fill_disps[n] = disp_collects[0];
				}
			}
			else {
				fill_disps[n] = disp_collects[disp_collects.size() / 2];
			}
		}
		for (auto n = 0u; n < trg_pixels.size(); n++) {
			auto& pix = trg_pixels[n];
			const int y = pix.first;
			const int x = pix.second;
			disp_ptr[y * width + x] = fill_disps[n];
		}
	}
}

void RemoveSpeckles(float* disparity_map, const int& width, const int& height,
	const int& diff_insame, const unsigned int& min_speckle_aera, const int& invalid_val)
{
	assert(width > 0 && height > 0);
	if (width < 0 || height < 0) {
		return;
	}

	// 定义标记像素是否访问的数组
	std::vector<bool> visited(int(width * height), false);
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			if (visited[i * width + j] || disparity_map[i * width + j] == invalid_val) {
				// 跳过已访问的像素及无效像素
				continue;
			}
			// 广度优先遍历，区域跟踪
			// 把连通域面积小于阈值的区域视差全设为无效值
			std::vector<std::pair<int, int>> vec;
			vec.emplace_back(i, j);
			visited[i * width + j] = true;
			unsigned int cur = 0;
			unsigned int next = 0;
			do {
				// 广度优先遍历区域跟踪	
				next = vec.size();
				for (unsigned int k = cur; k < next; k++) {
					const auto& pixel = vec[k];
					const int row = pixel.first;
					const int col = pixel.second;
					const auto& disp_base = disparity_map[row * width + col];
					// 8邻域遍历
					for (int r = -1; r <= 1; r++) {
						for (int c = -1; c <= 1; c++) {
							if (r == 0 && c == 0) {
								continue;
							}
							int rowr = row + r;
							int colc = col + c;
							if (rowr >= 0 && rowr < height && colc >= 0 && colc < width) {
								if (!visited[rowr * width + colc] &&
									(disparity_map[rowr * width + colc] != invalid_val) &&
									abs(disparity_map[rowr * width + colc] - disp_base) <= diff_insame) {
									vec.emplace_back(rowr, colc);
									visited[rowr * width + colc] = true;
								}
							}
						}
					}
				}
				cur = next;
			} while (next < vec.size());

			// 把连通域面积小于阈值的区域视差全设为无效值
			if (vec.size() < min_speckle_aera) {
				for (auto& pix : vec) {
					disparity_map[pix.first * width + pix.second] = invalid_val;
				}
			}
		}
	}
}


void MedianFilter(const float* in, float* out, const int& width, const int& height,
	const int wnd_size)
{
	const int radius = wnd_size / 2;
	const int size = wnd_size * wnd_size;

	// 存储局部窗口内的数据
	std::vector<float> wnd_data;
	wnd_data.reserve(size);

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			wnd_data.clear();

			// 获取局部窗口数据
			for (int r = -radius; r <= radius; r++) {
				for (int c = -radius; c <= radius; c++) {
					const int row = i + r;
					const int col = j + c;
					if (row >= 0 && row < height && col >= 0 && col < width) {
						wnd_data.push_back(in[row * width + col]);
					}
				}
			}
			// 排序
			std::sort(wnd_data.begin(), wnd_data.end());
			// 取中值
			out[i * width + j] = wnd_data[wnd_data.size() / 2];
		}
	}
}