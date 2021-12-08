#pragma once
#include <opencv2/opencv.hpp>	
#include <iostream>

using namespace std;
using namespace cv;
//constexpr auto Invalid_Float = std::numeric_limits<float>::infinity();


void LeftAndRightConsistency(float* leftDisp, float* rightDisp,float* lastDisp, int col, int row, float gate,
	vector<pair<int, int>>&occlusion, vector<pair<int, int>> &mismatch)
{
	 // ����һ���Լ��
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            // ��Ӱ���Ӳ�ֵ
			auto& lastvalue = lastDisp[i * col + j];
        	auto& disp = leftDisp[i * col + j];
			//if(disp == Invalid_Float){             // ����ط�Ҫ�ļ����Ӳ�ĳ���
			//	mismatches.emplace_back(i, j);
			//	continue;
			//}

            // �����Ӳ�ֵ�ҵ���Ӱ���϶�Ӧ��ͬ������
        	const auto col_right = static_cast<int>(j - disp + 0.5);  // Ϊʲô��0.5 ����  ��Ϊ������ǿ������ת��
            
        	if(col_right >= 0 && col_right < col) 
			{
                // ��Ӱ����ͬ�����ص��Ӳ�ֵ
                const auto& disp_r = rightDisp[i * col + col_right];
                
        		// �ж������Ӳ�ֵ�Ƿ�һ�£���ֵ����ֵ�ڣ�
        		if (abs(disp - disp_r) >= gate) {
					// �����ڵ�������ƥ����
					// ͨ����Ӱ���Ӳ��������Ӱ���ƥ�����أ�����ȡ�Ӳ�disp_rl
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

                    // ���Ӳ�ֵ��Ч
					//disp = 0;
					lastvalue = 0;
                }
				else
				{
					lastvalue = leftDisp[i * col + j];
				}
            }
            else{
                // ͨ���Ӳ�ֵ����Ӱ�����Ҳ���ͬ�����أ�����Ӱ��Χ��
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

	// �ڵ������غ���ƥ��������
	auto& occlusions = occlusion;
	auto& mismatches = mismatch;
	occlusions.clear();
	mismatches.clear();

	// ---����һ���Լ��
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			// ��Ӱ���Ӳ�ֵ
			auto& disp = leftDisp[i * width + j];
			if (disp == Invalid_Float) {
				mismatches.emplace_back(i, j);
				continue;
			}

			// �����Ӳ�ֵ�ҵ���Ӱ���϶�Ӧ��ͬ������
			const auto col_right = static_cast<int>(j - disp + 0.5);  // Ϊʲô��0.5 ���� 

			if (col_right >= 0 && col_right < width) {
				// ��Ӱ����ͬ�����ص��Ӳ�ֵ
				const auto& disp_r = rightDisp[i * width + col_right];

				// �ж������Ӳ�ֵ�Ƿ�һ�£���ֵ����ֵ�ڣ�
				if (abs(disp - disp_r) > threshold) {
					// �����ڵ�������ƥ����
					// ͨ����Ӱ���Ӳ��������Ӱ���ƥ�����أ�����ȡ�Ӳ�disp_rl
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

					// ���Ӳ�ֵ��Ч
					disp = Invalid_Float;
				}
			}
			else {
				// ͨ���Ӳ�ֵ����Ӱ�����Ҳ���ͬ�����أ�����Ӱ��Χ��
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

	// ����8������
	const float pi = 3.1415926f;
	float angle1[8] = { pi, 3 * pi / 4, pi / 2, pi / 4, 0, 7 * pi / 4, 3 * pi / 2, 5 * pi / 4 };
	float angle2[8] = { pi, 5 * pi / 4, 3 * pi / 2, 7 * pi / 4, 0, pi / 4, pi / 2, 3 * pi / 4 };
	float* angle = angle1;
	// ��������г̣�û�б�Ҫ������Զ������
	const int max_search_length = 1.0 * dispRange;

	float* disp_ptr = dispLeft;
	for (int k = 0; k < 3; k++) {
		// ��һ��ѭ�������ڵ������ڶ���ѭ��������ƥ����
		auto& trg_pixels = (k == 0) ? occlusion : mismatch;
		if (trg_pixels.empty()) {
			continue;
		}
		std::vector<float> fill_disps(trg_pixels.size());
		std::vector<std::pair<int, int>> inv_pixels;
		if (k == 2) {
			//  ������ѭ������ǰ����û�д���ɾ�������
			for (int i = 0; i < height; i++) {
				for (int j = 0; j < width; j++) {
					if (disp_ptr[i * width + j] == 0xffff) {
						inv_pixels.emplace_back(i, j);
					}
				}
			}
			trg_pixels = inv_pixels;
		}

		// ��������������
		for (auto n = 0u; n < trg_pixels.size(); n++) {
			auto& pix = trg_pixels[n];
			const int y = pix.first;
			const int x = pix.second;

			if (y == height / 2) {
				angle = angle2;
			}

			// �ռ�8���������������׸���Ч�Ӳ�ֵ
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

			// ������ڵ�������ѡ��ڶ�С���Ӳ�ֵ
			// �������ƥ��������ѡ����ֵ
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

	// �����������Ƿ���ʵ�����
	std::vector<bool> visited(int(width * height), false);
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			if (visited[i * width + j] || disparity_map[i * width + j] == invalid_val) {
				// �����ѷ��ʵ����ؼ���Ч����
				continue;
			}
			// ������ȱ������������
			// ����ͨ�����С����ֵ�������Ӳ�ȫ��Ϊ��Чֵ
			std::vector<std::pair<int, int>> vec;
			vec.emplace_back(i, j);
			visited[i * width + j] = true;
			unsigned int cur = 0;
			unsigned int next = 0;
			do {
				// ������ȱ����������	
				next = vec.size();
				for (unsigned int k = cur; k < next; k++) {
					const auto& pixel = vec[k];
					const int row = pixel.first;
					const int col = pixel.second;
					const auto& disp_base = disparity_map[row * width + col];
					// 8�������
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

			// ����ͨ�����С����ֵ�������Ӳ�ȫ��Ϊ��Чֵ
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

	// �洢�ֲ������ڵ�����
	std::vector<float> wnd_data;
	wnd_data.reserve(size);

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			wnd_data.clear();

			// ��ȡ�ֲ���������
			for (int r = -radius; r <= radius; r++) {
				for (int c = -radius; c <= radius; c++) {
					const int row = i + r;
					const int col = j + c;
					if (row >= 0 && row < height && col >= 0 && col < width) {
						wnd_data.push_back(in[row * width + col]);
					}
				}
			}
			// ����
			std::sort(wnd_data.begin(), wnd_data.end());
			// ȡ��ֵ
			out[i * width + j] = wnd_data[wnd_data.size() / 2];
		}
	}
}