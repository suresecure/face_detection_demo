// Face detection demo.
// Author: Robert (robert165 AT 163.com)
// Create: 2016-08-22
// Last modify: 2016-08-24
#include <stdio.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <iterator>
#include <algorithm>
#include <queue>

#include <time.h>
#include "opencv2/opencv.hpp"
//#include "opencv2/highgui/highgui.hpp"
//#include "boost/filesystem.hpp" // Donot use boost.

#include "facedetect-dll.h"
#pragma comment(lib,"libfacedetect.lib")
#include <Windows.h>
#include <imagehlp.h>
#pragma comment(lib,"imagehlp.lib")

//namespace fs = ::boost::filesystem;

using namespace cv;
using namespace std;

/*
std::vector<std::string> &split(const std::string &s, char delim,
                                std::vector<std::string> &elems) {
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, delim)) {
        elems.push_back(item);
    }
    return elems;
}

std::vector<std::string> split(const std::string &s, char delim) {
    std::vector<std::string> elems;
    split(s, delim, elems);
    return elems;
}

void getAllFiles(const fs::path &root, const string &ext, vector<fs::path> &ret) {
    if (!fs::exists(root) || !fs::is_directory(root))
        return;

    fs::recursive_directory_iterator it(root);
    fs::recursive_directory_iterator endit;

    while (it != endit) {
        if (fs::is_regular_file(*it) && it->path().extension() == ext)
            ret.push_back(it->path());
        ++it;
    }
}
*/

/* // "localtime" is not safe in windows
// Generate a new file name in the format "yyyymmdd_hh_mm_ss_RAND".
string getNewFileName(const string ext) {
    srand((int) time(0));
    int num = rand()%1000;
    struct tm *p;
    time_t second;
    time(&second);
    p = localtime(&second);
    char buf[100] = {'\0'};
    sprintf(buf, "%04d%02d%02d_%02d%02d%02d_%03d", 1900+p->tm_year,
            1+p->tm_mon, p->tm_mday,
            p->tm_hour, p->tm_min, p->tm_sec, num);
    return string(buf)+ext;
}
*/

// Read line one by one from the specified txt file, and store in a string vector.
vector<string> readLines(const string & file) {
    ifstream file_list(file.c_str());
    string line;
    vector<string> ret;
    while (getline(file_list, line)) {
        ret.push_back(line);
    }
    return ret;
}

// Expand the detect face rect.
Rect expandRect(const Rect & rect, const Size & size, const float & ratio) {
    int width = (int)(rect.width * ratio);
    int height = (int)(rect.height * ratio);
    int x = rect.x - (int)((width - rect.width) / 2);
    int y = rect.y - (int)((height - rect.height) / 2);

    if (x < 0) {
        width += x;
        x = 0;
    }
    if (y < 0) {
        height += y;
        y = 0;
    }
    if (x + width >= size.width) {
        width = size.width - 1 - x;
    }
    if (y + height >= size.height) {
        height = size.height - 1 - y;
    }
    return Rect(x, y, width, height);
}

// Find the max rect.
Rect  maxRect(const vector<Rect> & rects) {
	int maxArea = 0;
	int ind_max = -1;
	for (int i = 0; i < rects.size(); i++) {
		int area = rects[i].width * rects[i].height;
		if (area > maxArea) {
			maxArea = area;
			ind_max = i;
		}
	}
	if (ind_max < 0)
		return Rect();
	return rects[ind_max];
}

// IOU of two rects.
float IOU(const Rect & r1, const Rect &  r2) {
	if (r1.area() <= 0 || r2.area() <= 0)
		return 0;

	Rect intersect = r1 & r2;
	return (float)intersect.area() / (r1.area() + r2.area() - intersect.area());
}

// Merge two list of rects (assume rects in the same list do not intersect). 
// Keep the larger one when two rects intersect.
// th_iou is the iou threshold to recognize intersected rects.
int mergeDetectFaceRects(vector<Rect> & r1, const vector<Rect> & r2, 
	const float th_iou, vector<int> & n1, const vector<int> &n2) {
	if (0 == r1.size()) {
		r1 = r2;
		n1 = n2;
		return r1.size();
	}

	if (0 == r2.size())
		return r1.size();

	vector<Rect> r;
	vector <int> n;
	bool * ind = new bool[r2.size()]; // Is the rect in r2 still valid.
	for (int j = 0; j < r2.size(); j++)
		ind[j] = true;

	// Find and add valid rects in r1.
	for (int i = 0; i < r1.size(); i++) {
		bool valid = true;
		for (int j = 0; j < r2.size() && ind[j]; j++){
			float iou = IOU(r1[i], r2[j]);
			if (iou > th_iou) { // Rects from r1 and r2 intersect.
				// Use the larger rect (rect from r1 when equal size).
				if (r1[i].area() < r2[j].area()) { 
					valid = false;
					break;
				}
				else
					ind[j] = false;
			}
		}
		if (valid) {
			r.push_back(r1[i]);
			n.push_back(n1[i]);
		}
	}

	// Add valid rects in r2.
	for (int j = 0; j < r2.size() && ind[j]; j++)  {
		r.push_back(r2[j]);
		n.push_back(n2[j]);
	}

	delete[] ind;
	r1 = r;
	return r1.size();
}

int mergeDetectFaceRects(vector<Rect> & r1, const vector<Rect> & r2, const float th_iou) {
	return mergeDetectFaceRects(r1, r2, th_iou, vector<int>(r1.size()), vector<int>(r2.size()));
}

int detectFace(Mat & frame, vector<Rect> & face_rects, vector<int> & face_neighbors, float & iou_multi_faces){
	if (frame.cols <= 0 || frame.rows <= 0 || frame.channels() > 1){
		cerr << "Image frame format error for face detection. It should be a gray image." << endl;
		return 0;
	}

	int * pResults = NULL;
	///////////////////////////////////////////
	// new frontal face detection 
	// it can detect faces with bad illumination.
	//////////////////////////////////////////
	//!!! The input image must be a gray one (single-channel)
	//!!! DO NOT RELEASE pResults !!!
	long time_facedetect_frontal_tmp_start = clock();
	pResults = facedetect_frontal_tmp((unsigned char*)(frame.ptr(0)),
		frame.cols, frame.rows, frame.step,
		1.2f, 5, 24);
	cout << "facedetect_frontal_tmp: " << (clock() - time_facedetect_frontal_tmp_start) << " ms" << endl;
	//printf("%d faces detected.\n", (pResults ? *pResults : 0));
	//vector<Rect> face_rects;
	//vector<int> face_neighbors;
	for (int j = 0; j < (pResults ? *pResults : 0); j++)
	{
		short * p = ((short*)(pResults + 1)) + 6 * j;
		face_rects.push_back(Rect(p[0], p[1], p[2], p[3]));
		int neighbor = p[4];
		face_neighbors.push_back(neighbor);
	}
	// "facedetect_frontal_tmp()" could miss some side view face. 
	// Combine with other detection methods.

	// frontal face detection 
	// it's fast, but cannot detect side view face
	long time_facedetect_frontal_start = clock();
	pResults = facedetect_frontal((unsigned char*)(frame.ptr(0)),
		frame.cols, frame.rows, frame.step,
		1.2f, 3, 24);
	cout << "facedetect_frontal: " << (clock() - time_facedetect_frontal_start) << " ms" << endl;
	vector<Rect> face_rects_frontal;
	vector <int> neighbors_frontal;
	for (int j = 0; j < (pResults ? *pResults : 0); j++)
	{
		short * p = ((short*)(pResults + 1)) + 6 * j;
		face_rects_frontal.push_back(Rect(p[0], p[1], p[2], p[3]));
		int neighbor = p[4];
		neighbors_frontal.push_back(neighbor);
	}
	mergeDetectFaceRects(face_rects, face_rects_frontal, iou_multi_faces,
		face_neighbors, neighbors_frontal);

	/*
	// multiview face detection 
	// it can detect side view faces, but slower than facedetect_frontal().
	long time_facedetect_multiview_start = clock();
	pResults = facedetect_multiview((unsigned char*)(frame.ptr(0)),
		frame.cols, frame.rows, frame.step,
		1.2f, 5, 24);
	cout << "facedetect_multiview: " << (clock() - time_facedetect_multiview_start) << " ms" << endl;
	vector<Rect> face_rects_multiview;
	vector <int> neighbors_multiview;
	for (int j = 0; j < (pResults ? *pResults : 0); j++)
	{
		short * p = ((short*)(pResults + 1)) + 6 * j;
		face_rects_multiview.push_back(Rect(p[0], p[1], p[2], p[3]));
		int neighbor = p[4];
		neighbors_multiview.push_back(neighbor);
	}
	mergeDetectFaceRects(face_rects, face_rects_multiview, iou_multi_faces,
		face_neighbors, neighbors_multiview);
	*/

	/*
	// multiview_reinforce face detection 
	// it can detect side view faces, better but slower than facedetect_multiview().
	long time_facedetect_multiview_reinforce_start = clock();
	pResults = facedetect_multiview_reinforce((unsigned char*)(frame.ptr(0)),
		frame.cols, frame.rows, frame.step,
		1.2f, 5, 24);
	cout << "facedetect_multiview_reinforce: " <<
		(clock() - time_facedetect_multiview_reinforce_start) << " ms" << endl;
	vector<Rect> face_rects_multiview_reinforce;
	vector <int> neighbors_multiview_reinforce;
	for (int j = 0; j < (pResults ? *pResults : 0); j++)
	{
		short * p = ((short*)(pResults + 1)) + 6 * j;
		face_rects_multiview_reinforce.push_back(Rect(p[0], p[1], p[2], p[3]));
		int neighbor = p[4];
		neighbors_multiview_reinforce.push_back(neighbor);
	}
	mergeDetectFaceRects(face_rects, face_rects_multiview_reinforce, iou_multi_faces,
		face_neighbors, neighbors_multiview_reinforce);
	*/

	return face_rects.size();
}

int main(int argc, char **argv) {

	// Parameter to be set.
	float resize_factor = 0.25; // Resize frame before detection to speed up.
	int num_previous_frames_for_tracking = 3; // Number of previous frames to be check for face tracking.
	float iou_previous_rect_for_tracking = 0.75; // IOU threshold to recognize same faces of two frames. 
	float iou_multi_faces = 0.3; // IOU threshold for multi faces appear in the same image.
	string save_folder_name("libfd_face_image"); // Folder to save detect face images.
	char delim = '\\'; // Windows path style.

    //cout<<"Using OpenCV version " << CV_VERSION << std::endl;
    //cout<<getBuildInformation();

	vector<string> video_list;
	if (1 < argc) {
		string filename = argv[1];
		if (".txt" == filename.substr(filename.length() - 4))
			video_list = readLines(filename);
		else
			video_list.push_back(filename);
	}
	if (0 == argc) {
		string video_list_file = "../data/path_video_win.txt";
		video_list = readLines(video_list_file);
	}

    long time_start = clock();
    int total_frame = 0;
    for(int N = 0; N < video_list.size(); N++) {
        long time_video_start = clock();
        string video = video_list.at(N);
        cout<<"-------------------------------------------------------------"<<endl;
        cout<<"VIDEO #"<<N<<": "<<video<<endl;

        // get a handle to the camera
        VideoCapture cap(video);
        //VideoCapture cap(0);
        if (!cap.isOpened()) {
			cerr<<"Camera cannot be opened or video file broken."<<endl;
			continue;
        }

		// Get face image save path.
		/*fs::path save_folder(video.substr(0, video.length() - 4));
		fs::path video_name = save_folder.filename();
		save_folder = save_folder.parent_path().parent_path();
		save_folder /= fs::path(save_folder_name);
		save_folder /= video_name;
		//cout << "Save folder: " << save_folder << endl;
		fs::create_directories(save_folder);
		*/
		// Donot use boost::filesystem::path
		string save_folder = video.substr(0, video.length() - 4);
		for (string::iterator iter = save_folder.begin(); iter != save_folder.end(); iter++) {
			if ('\\' == *iter || '\/' == *iter)
				*iter = delim;
		}
		string video_name = save_folder.substr(save_folder.find_last_of(delim)+1);
		save_folder = save_folder.substr(0, save_folder.find_last_of(delim));
		save_folder = save_folder.substr(0, save_folder.find_last_of(delim));
		save_folder = save_folder + delim + save_folder_name + delim + video_name + delim;
		cout << "Save folder: " << save_folder << endl;
		MakeSureDirectoryPathExists(save_folder.c_str());

		Mat frame, frame_clone;
        int count_frame = cap.get(CV_CAP_PROP_FRAME_COUNT);
        int width = cap.get(CV_CAP_PROP_FRAME_WIDTH);
        int height = cap.get(CV_CAP_PROP_FRAME_HEIGHT);
		int fps = cap.get(CV_CAP_PROP_FPS);
		// cap.set(CV_CAP_PROP_POS_MSEC, (9 * 60 + 20) * 1000); // "loby" video, seek to multi-face scenaria.
        //int count_frame = 1000;
        cout<<"Frame count: "<<count_frame<<endl;
        cout<<"Frame size: "<<width<<"X"<<height<<endl;
		cout << "FPS: " << fps << endl;

		width = width * resize_factor;
		height = height * resize_factor;

		int count_face = 0;  // Number of detected faces in the current video.
		int count_empty_frame = 0; // Number of empty frames.
		// Detection results of the previous frames. Each queue to track a face.
		vector<deque <Rect> > previous_rects;
		Rect emptyRect;
		for (int i = 0; i < count_frame; i++) {
			long time_frame_start = clock();
			cap >> frame;
			cout << "Frame #" << i << ": " << frame.cols << "X" << frame.rows;
			cout << ", load in: " << (clock() - time_frame_start) << " ms" << endl;
			frame_clone = frame.clone();
			if (frame.rows*frame.cols <= 0)
			{
				cerr << " ERROR: frame is empty!" << endl;
				count_empty_frame++;
				continue;
			}
			resize(frame, frame, Size(width, height));

			// Detect face using libfacedetection.
			cvtColor(frame, frame, CV_RGB2GRAY);
			vector<Rect> face_rects, rects_show;
			vector<int> face_neighbors;
			int num_faces = detectFace(frame, face_rects, face_neighbors, iou_multi_faces);
			count_face += num_faces;
			if (0 == num_faces) {
				cout << "No face detected!" << endl;
			}				
			
			// Add empty rect to all the face tracking queue.
			for (vector< deque<Rect> >::iterator iter = previous_rects.begin();
				iter != previous_rects.end(); iter++)
				iter->push_back(emptyRect);

			//Examine each rect, save new detection to the disk.
			for (int ind_face = 0; ind_face < face_rects.size(); ind_face++) {
				Rect face_rect = face_rects[ind_face];

				// Resize rect to the origin frame size.
				face_rect = Rect(int(face_rect.x / resize_factor), int(face_rect.y / resize_factor),
					int(face_rect.width / resize_factor), int(face_rect.height / resize_factor));
				face_rect = expandRect(face_rect, frame_clone.size(), 1); // Ensure rect is not overranged.
				// Rect for video display.
				Rect rect_show(face_rect.x / 3, face_rect.y / 3, face_rect.width / 3, face_rect.height / 3);
				rects_show.push_back(rect_show);

				// The detection rect is tightly to the face. Expand the rect.
				Rect detect_face_rect = expandRect(face_rect, frame_clone.size(), 3);
				cout << ind_face+1 << "/" << face_rects.size() << ": " << face_rect << endl;

				// Face tracking to avoid saving of duplicate faces.
				bool is_dup = false;
				for (vector< deque<Rect> >::iterator iter = previous_rects.begin();
					iter != previous_rects.end(); iter++) {
					for (int j = 0; j < iter->size(); j++) {
						float iou = IOU(detect_face_rect, iter->at(j));
						if (iou > iou_previous_rect_for_tracking) {
							is_dup = true;
							// Substitue empty rect in the end of the queue to the detect face rect.
							iter->at(j) = detect_face_rect;
							break;
						}
					}
				}
				if (!is_dup) { // Write to the disk if it is a new face.
					Mat detect_face = frame_clone(detect_face_rect);
					/*fs::path save_image_path = save_folder;
					save_image_path /= fs::path(to_string(i) + ".jpg");
					//cout<<save_image_path.string()<<endl;
					imwrite(save_image_path.string(), detect_face);
					*/
					string save_image_path = save_folder + to_string(i) + ".jpg";
					//cout<<save_image_path.string()<<endl;
					imwrite(save_image_path, detect_face);
					// Add a new face tracking queue
					deque<Rect> newQueue;
					newQueue.push_back(detect_face_rect);
					previous_rects.push_back(newQueue);
				}

			}

			// Update face tracking data structure.
			for (vector< deque<Rect> >::iterator iter = previous_rects.begin();
				iter != previous_rects.end();) {
				// Track across at most "num_previous_frames_for_tracking" frames.
				if (iter->size() > num_previous_frames_for_tracking)
					iter->pop_front();
				// Delete lost face (all rect are empty in the queue).
				bool valid = false;
				for (deque<Rect>::iterator it = iter->begin(); it != iter->end(); it++) {
					if (it->area() > 0) {
						valid = true;
						break;
					}
				}
				if (valid)
					iter++;
				else
					iter = previous_rects.erase(iter);
			}

			// Video display.
			resize(frame_clone, frame_clone, Size(frame_clone.cols / 3, frame_clone.rows/3));
			for (vector< Rect >::iterator iter = rects_show.begin();
				iter != rects_show.end() && iter->area() > 0; iter++) {
				rectangle(frame_clone, *iter, Scalar(255, 0, 255));
			}
            imshow("Face Video", frame_clone);
            if ( 27 == waitKey(1))
                return 0;

			cout <<"frame: " << (clock() - time_frame_start) << " ms" << endl;
        }
        cap.release();
		count_frame -= count_empty_frame;
		total_frame += count_frame;

        long time_video_end = clock();
        cout<<"Processed in "<<(float)(time_video_end - time_video_start)/1000.0<<" sec., "
               <<(float)(time_video_end - time_video_start)/count_frame
               <<" ms per "<<width<<"X"<<height<<" frame."<<endl;
        cout<<count_face<<"/"<<count_frame<<" faces detected."<<endl;
    }

    long time_finish = clock();
    cout<<"Processed "<<video_list.size()<<" videos with "<<(float) total_frame/1000<<" K frames"
          <<"in "<<float(time_finish - time_start)/1000.0/3600.0<<" hours"<<endl;

    return 0;
}
