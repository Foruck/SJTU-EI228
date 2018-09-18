#include <opencv2\opencv.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <iostream>
#include <string>
#include <vector>
#include "BluetoothTrans.h"
#include <cmath>
using namespace cv;
using namespace std;
struct location {
	int x, y;
	location() = default;
	location(int a, int b) :x(a), y(b) {}
	location operator=(location& other) {
		if (&other == this) return *this;
		x = other.x; y = other.y;
		return *this;
	}
};
Point2f pointsForTrans[4],newPoints[4];
Point2f pointsForTarget[2];
int tot(0);
bool glag(false);
Mat transmat;
int a[20][20];
vector<int> sx, sy;
vector<char> spin;
const int channels[1] = { 0 };
const int histsize[1] = { 30 };
float range[2] = { 0, 180 };
const float *ranges[1] = { range };
MatND hist1, hist2;
location get_loc(Rect& target1, Rect& target2) {
	int x0, y0, x, y;
	x0 = target1.x + target1.width / 2;
	y0 = target1.y + target1.height / 2;
	x = target2.x + target2.width / 2;
	y = target2.y + target2.height / 2;
	x0 = (x0 + x) / 2;
	y0 = (y0 + y) / 2;
	return location(y0, x0);
}

void OnMouse(int mouseevent, int x, int y, int flag, void *param) {
	if (tot != 4 && mouseevent == CV_EVENT_LBUTTONDOWN) {
		pointsForTrans[tot++] = Point2f(x, y); 
		cout << x << ' ' << y << endl;
		return;
	}
	if (tot>=4){
		newPoints[0] = Point2f(0, 0);
		newPoints[1] = Point2f(400, 0);
		newPoints[2] = Point2f(0, 400);
		newPoints[3] = Point2f(400, 400);
		transmat = getPerspectiveTransform(pointsForTrans, newPoints);
		//cout << "True" << endl;
		glag = true;
		return;
	}
}

void OnMouseForTargeting(int mouseevent, int x, int y, int flag, void*param) {
	if (tot != 2 && mouseevent == CV_EVENT_LBUTTONDOWN) {
		pointsForTarget[tot++] = Point2f(x, y);
		return;
	}
	if (tot >= 2) { glag = true; return; }
}

void Get_map(Mat &src) {
	int i, j;
	CvScalar s;
	IplImage tmp = IplImage(src);

	for (i = 0; i <= 5; i++)
		for (j = 0; j <= 5; j++)
			a[i][j] = 1;

	for (i=50;i<400;i+=100)
		for (j = 50; j < 400; j += 100) {
			s = cvGet2D(&tmp, j, i);
			if (s.val[0] == 255) a[j / 100 + 1][i / 100 + 1] = 0;
			else a[j / 100 + 1][i / 100 + 1] = 1;
		}
}

bool Find_path(int x, int y, int x0, int y0) {
	if ((x == 1 || x == 4 || y == 0 || y == 4) && (x != x0 && y != y0)) return true;
	bool b = false;
	if (!a[x + 1][y]) {
		a[x + 1][y] = 1; sx.push_back(x + 1); sy.push_back(y);
		b=Find_path(x + 1, y, x0, y0);
		if (b) return b;
		a[x + 1][y] = 0; sx.pop_back(); sy.pop_back();
	}
	if (!a[x - 1][y]) {
		a[x - 1][y] = true; sx.push_back(x - 1); sy.push_back(y);
		b=Find_path(x - 1, y, x0, y0);
		if (b) return b;
		a[x - 1][y] = 0; sx.pop_back(); sy.pop_back();
	}
	if (!a[x][y + 1]) {
		a[x][y + 1] = true; sx.push_back(x); sy.push_back(y + 1);
		b=Find_path(x, y + 1, x0, y0);
		if (b) return b;
		a[x][y + 1] = 0; sx.pop_back(); sy.pop_back();
	}
	if (!a[x][y - 1]) {
		a[x][y - 1] = true; sx.push_back(x); sy.push_back(y - 1);
		b = Find_path(x, y - 1, x0, y0);
		if (b) return b;
		a[x][y - 1] = 0; sx.pop_back(); sy.pop_back();
	}
	return false;
}

void Find_car(VideoCapture& cam, Rect& target1, Rect& target2) {
	Mat ori, tmp, tmp_hsv, ans2;
	IplImage *p, *q, *r;
	while (!glag) {
		cam >> ori;
		warpPerspective(ori, tmp, transmat, Size(400, 400));
		cvtColor(tmp, tmp_hsv, CV_BGR2HSV);
		inRange(tmp_hsv, Scalar(35, 43, 46), Scalar(99, 255, 255), ans2);
		imshow("tmp", tmp);
		imshow("threshold", ans2);
		setMouseCallback("threshold", OnMouseForTargeting);
		char c = (char)waitKey(5);
		if (c == 27) break;
	}
	target1 = Rect(pointsForTarget[0], pointsForTarget[1]);
	tot = 0; glag = false;
	while (!glag) {
		cam >> ori;
		warpPerspective(ori, tmp, transmat, Size(400, 400));
		cvtColor(tmp, tmp_hsv, CV_BGR2HSV);
		inRange(tmp_hsv, Scalar(100, 43, 46), Scalar(155, 255, 255), ans2);
		imshow("tmp", tmp);
		imshow("threshold", ans2);
		setMouseCallback("threshold", OnMouseForTargeting);
		char c = (char)waitKey(5);
		if (c == 27) break;
	}
	target2 = Rect(pointsForTarget[0], pointsForTarget[1]);
	destroyWindow("threshold");
	destroyWindow("tmp");
	//get target2
}

void Sample(VideoCapture& cam) {
	int t;
	char ch[5][50];
	Mat tmp;
	for (int i = 0; i < 4; i++) {
		cin >> ch[i];
		cam >> tmp;
		imwrite(ch[i], tmp);
	}
}

void getHist() {
	vector<Mat> chs;
	Mat cap = imread("sample1.png"), caphsv = Mat(cap.size(), 8, 3);
	cvtColor(cap, caphsv, CV_BGR2HSV);
	split(caphsv, chs);
	calcHist(&chs[0], 1, channels, Mat(), hist1, 1, histsize, ranges);
	//hist1
	cap = imread("sample2.png");
	cvtColor(cap, caphsv, CV_BGR2HSV);
	split(caphsv, chs);
	calcHist(&chs[0], 1, channels, Mat(), hist2, 1, histsize, ranges);
	//hist2
	return;
}

void get_statue() {
	int i, j, k;
	spin.push_back('L');
	cout << "L";
	for (i = 1; i < sx.size(); i++) {
		switch (spin[i - 1]) {
		case 'U':
			if (sy[i] == sy[i - 1] && sx[i] < sx[i - 1]) spin.push_back('U');
			if (sx[i] == sx[i - 1] && sy[i] < sy[i - 1]) spin.push_back('L');
			if (sx[i] == sx[i - 1] && sy[i] > sy[i - 1]) spin.push_back('R');
			break;
		case 'L':
			if (sx[i] == sx[i - 1] && sy[i] < sy[i - 1]) spin.push_back('L');
			if (sy[i] == sy[i - 1] && sx[i] < sx[i - 1]) spin.push_back('U');
			if (sy[i] == sy[i - 1] && sx[i] > sx[i - 1]) spin.push_back('D');
			break;
		case 'R':
			if (sx[i] == sx[i - 1] && sy[i] > sy[i - 1]) spin.push_back('R');
			if (sy[i] == sy[i - 1] && sx[i] < sx[i - 1]) spin.push_back('U');
			if (sy[i] == sy[i - 1] && sx[i] > sx[i - 1]) spin.push_back('D');
			break;
		case 'D':
			if (sy[i] == sy[i - 1] && sx[i] > sx[i - 1]) spin.push_back('D');
			if (sx[i] == sx[i - 1] && sy[i] < sy[i - 1]) spin.push_back('L');
			if (sx[i] == sx[i - 1] && sy[i] > sy[i - 1]) spin.push_back('R');
			break;
		}
		cout << spin[i];
	}
	cout << endl;
}

void init(VideoCapture& cam, Rect& target1, Rect& target2) {
	Mat origin, transformed, greyed, thersholded;
	int x, y, l;
	location loc;
	//init
	
	while (!glag) {
		cam >> origin;
		imshow("Origin", origin);
		setMouseCallback("Origin", OnMouse);
		char c = (char)waitKey(5);
		if (c == 27) break;
	}
	cam >> origin;
	warpPerspective(origin, transformed, transmat, Size(400, 400));
	cvtColor(transformed, greyed, CV_BGR2GRAY);
	threshold(greyed, thersholded, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
	Get_map(thersholded);
	while (true) {
		cam >> origin;
		warpPerspective(origin, transformed, transmat, Size(400, 400));
		imshow("Transform", transformed);
		cvtColor(transformed, greyed, CV_BGR2GRAY);
		threshold(greyed, thersholded, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
		imshow("Threshold", thersholded);
		char c = (char)waitKey(5);
		if (c == 27) break;
	}
	destroyWindow("Threshold");
	destroyWindow("Origin");
	destroyWindow("Transform");
	//get_map
	
	//getHist();
	glag = false; tot = 0;
	Find_car(cam, target1, target2);
	//get_car
	loc = get_loc(target1, target2);
	x = loc.x / 100 + 1; y = loc.y / 100 + 1;
	a[x][y] = 1;
	sx.push_back(x); sy.push_back(y);
	Find_path(x, y, x, y);
	//get_path
	for (int i = 0; i < sx.size(); i++)
		cout << sx[i] << ' ' << sy[i] << endl;
	get_statue();
	cout << "Init Over!" << endl;
}

bool check(location& loc, int x, int y) {
	cout << 'Z';
	if (loc.x <= x * 100 - 10 && loc.x >= (x - 1) * 100 + 10 && loc.y <= y * 100 - 10 && loc.y >= (y - 1) * 100 + 10) return true;
	return false;
}

bool check_direction(Rect& target1, Rect& target2, char c) {
	int x1, y1, x2, y2;
	x1 = target1.x + target1.width / 2; y1 = target1.y + target1.height / 2;
	x2 = target2.x + target2.width / 2; y2 = target2.y + target2.height / 2;
	cout << x1 << ' ' << y1 << ' ' << x2 << ' ' << y2 << endl;
	if (x1 - x2 < -22 && abs(y1 - y2) <= 20 && c == 'L') return true;
	if (x1 - x2 > 22 && abs(y1 - y2) <= 20 && c == 'R') return true;
	if (y1 - y2 < -22 && abs(x1 - x2) <= 20 && c == 'U') return true;
	if (y1 - y2 >22 && abs(x1 - x2) <= 20 && c == 'D') return true;
	return false;
}

int main(){
	Mat result1, result2, backproject, cvt_hsv, origin, transformed, thresholded, result;
	IplImage *r, *p, *q;
	int l, time = 0;
	VideoCapture cam(1);
	Rect target1, target2;
	location loc;
	int x;
	//Pause();
	tot = 0; glag = false; 
	init(cam, target1, target2);
	loc = get_loc(target1, target2);
	l = 0;
	while (l < sx.size() - 1) {
		tot = 0;
		switch (spin[l]) {
		case 'U':
			if (spin[l + 1] == 'L') {
				while (!check_direction(target1,target2,'L')) {
					Left();
					Sleep(500);
					cam >> origin;
					warpPerspective(origin, transformed, transmat, Size(400, 200));
					imshow("Monitor", transformed);
					cvtColor(transformed, cvt_hsv, CV_BGR2HSV);
					inRange(cvt_hsv, Scalar(35, 43, 46), Scalar(99, 255, 255), thresholded);
					meanShift(thresholded, target1, TermCriteria(3, 10, 1));
					cvtColor(transformed, cvt_hsv, CV_BGR2HSV);
					inRange(cvt_hsv, Scalar(100, 43, 46), Scalar(155, 255, 255), thresholded);
					meanShift(thresholded, target2, TermCriteria(3, 10, 1));
					loc = get_loc(target1, target2);
					char c = (char)waitKey(5);
					if (c == 27) break;
				}
			}
			if (spin[l + 1] == 'R') {
				while (!check_direction(target1, target2, 'R')) {
					Right();
					Sleep(500);
					cam >> origin;
					warpPerspective(origin, transformed, transmat, Size(400, 200));
					imshow("Monitor", transformed);
					cvtColor(transformed, cvt_hsv, CV_BGR2HSV);
					inRange(cvt_hsv, Scalar(35, 43, 46), Scalar(99, 255, 255), thresholded);
					meanShift(thresholded, target1, TermCriteria(3, 10, 1));
					cvtColor(transformed, cvt_hsv, CV_BGR2HSV);
					inRange(cvt_hsv, Scalar(100, 43, 46), Scalar(155, 255, 255), thresholded);
					meanShift(thresholded, target2, TermCriteria(3, 10, 1));
					loc = get_loc(target1, target2);
					char c = (char)waitKey(5);
					if (c == 27) break;
				}
			}
			break;
		case 'D':
			if (spin[l + 1] == 'L') {
				while (!check_direction(target1, target2, 'L')) {
					Right();
					Sleep(500);
					cam >> origin;
					warpPerspective(origin, transformed, transmat, Size(400, 200));
					imshow("Monitor", transformed);
					cvtColor(transformed, cvt_hsv, CV_BGR2HSV);
					inRange(cvt_hsv, Scalar(35, 43, 46), Scalar(99, 255, 255), thresholded);
					meanShift(thresholded, target1, TermCriteria(3, 10, 1));
					cvtColor(transformed, cvt_hsv, CV_BGR2HSV);
					inRange(cvt_hsv, Scalar(100, 43, 46), Scalar(155, 255, 255), thresholded);
					meanShift(thresholded, target2, TermCriteria(3, 10, 1));
					loc = get_loc(target1, target2);
					char c = (char)waitKey(5);
					if (c == 27) break;
				}
			}
			if (spin[l + 1] == 'R') {
				while (!check_direction(target1, target2, 'R')) {
					Left();
					Sleep(500);
					cam >> origin;
					warpPerspective(origin, transformed, transmat, Size(400, 200));
					imshow("Monitor", transformed);
					cvtColor(transformed, cvt_hsv, CV_BGR2HSV);
					inRange(cvt_hsv, Scalar(35, 43, 46), Scalar(99, 255, 255), thresholded);
					meanShift(thresholded, target1, TermCriteria(3, 10, 1));
					cvtColor(transformed, cvt_hsv, CV_BGR2HSV);
					inRange(cvt_hsv, Scalar(100, 43, 46), Scalar(155, 255, 255), thresholded);
					meanShift(thresholded, target2, TermCriteria(3, 10, 1));
					loc = get_loc(target1, target2);
					char c = (char)waitKey(5);
					if (c == 27) break;
				}
			}
			break;
		case 'L':
			if (spin[l + 1] == 'U') {
				while (!check_direction(target1, target2, 'U')) {
					Right();
					Sleep(500);
					cam >> origin;
					warpPerspective(origin, transformed, transmat, Size(400, 200));
					imshow("Monitor", transformed);
					cvtColor(transformed, cvt_hsv, CV_BGR2HSV);
					inRange(cvt_hsv, Scalar(35, 43, 46), Scalar(99, 255, 255), thresholded);
					meanShift(thresholded, target1, TermCriteria(3, 10, 1));
					cvtColor(transformed, cvt_hsv, CV_BGR2HSV);
					inRange(cvt_hsv, Scalar(100, 43, 46), Scalar(155, 255, 255), thresholded);
					meanShift(thresholded, target2, TermCriteria(3, 10, 1));
					loc = get_loc(target1, target2);
					char c = (char)waitKey(5);
					if (c == 27) break;
				}
			}
			if (spin[l + 1] == 'D') {
				while (!check_direction(target1, target2, 'D')) {
					Left();
					Sleep(500);
					cam >> origin;
					warpPerspective(origin, transformed, transmat, Size(400, 200));
					imshow("Monitor", transformed);
					cvtColor(transformed, cvt_hsv, CV_BGR2HSV);
					inRange(cvt_hsv, Scalar(35, 43, 46), Scalar(99, 255, 255), thresholded);
					meanShift(thresholded, target1, TermCriteria(3, 10, 1));
					cvtColor(transformed, cvt_hsv, CV_BGR2HSV);
					inRange(cvt_hsv, Scalar(100, 43, 46), Scalar(155, 255, 255), thresholded);
					meanShift(thresholded, target2, TermCriteria(3, 10, 1));
					loc = get_loc(target1, target2);
					char c = (char)waitKey(5);
					if (c == 27) break;
				}
			}
			break;
		case 'R':
			if (spin[l + 1] == 'D') {
				while (!check_direction(target1, target2, 'D')) {
					Right();
					Sleep(500);
					cam >> origin;
					warpPerspective(origin, transformed, transmat, Size(400, 200));
					imshow("Monitor", transformed);
					cvtColor(transformed, cvt_hsv, CV_BGR2HSV);
					inRange(cvt_hsv, Scalar(35, 43, 46), Scalar(99, 255, 255), thresholded);
					meanShift(thresholded, target1, TermCriteria(3, 10, 1));
					cvtColor(transformed, cvt_hsv, CV_BGR2HSV);
					inRange(cvt_hsv, Scalar(100, 43, 46), Scalar(155, 255, 255), thresholded);
					meanShift(thresholded, target2, TermCriteria(3, 10, 1));
					loc = get_loc(target1, target2);
					char c = (char)waitKey(5);
					if (c == 27) break;
				}
			}
			if (spin[l + 1] == 'U') {
				while (!check_direction(target1, target2, 'U')) {
					Left();
					Sleep(500);
					cam >> origin;
					warpPerspective(origin, transformed, transmat, Size(400, 200));
					imshow("Monitor", transformed);
					cvtColor(transformed, cvt_hsv, CV_BGR2HSV);
					inRange(cvt_hsv, Scalar(35, 43, 46), Scalar(99, 255, 255), thresholded);
					meanShift(thresholded, target1, TermCriteria(3, 10, 1));
					cvtColor(transformed, cvt_hsv, CV_BGR2HSV);
					inRange(cvt_hsv, Scalar(100, 43, 46), Scalar(155, 255, 255), thresholded);
					meanShift(thresholded, target2, TermCriteria(3, 10, 1));
					loc = get_loc(target1, target2);
					char c = (char)waitKey(5);
					if (c == 27) break;
				}
			}
			break;
		}
		cout << l << "_Changing_Statue_Ends_" << spin[l] << spin[l + 1] << endl;
		tot = 0;
		while (!check(loc, sx[l + 1], sy[l + 1])) {
			Forward();
			cout << 'A';
			Sleep(1000);
			cout << 'B';
			cam >> origin;
			cout << 'C';
			warpPerspective(origin, transformed, transmat, Size(400, 400));
			imshow("Monitor", transformed);
			cvtColor(transformed, cvt_hsv, CV_BGR2HSV);
			inRange(cvt_hsv, Scalar(35, 43, 46), Scalar(99, 255, 255), thresholded);
			meanShift(thresholded, target1, TermCriteria(3, 10, 1));
			imshow("Thresholded", thresholded);
			cout << 'D';
			cvtColor(transformed, cvt_hsv, CV_BGR2HSV);
			inRange(cvt_hsv, Scalar(100, 43, 46), Scalar(155, 255, 255), thresholded);
			meanShift(thresholded, target2, TermCriteria(3, 10, 1));
			cout << 'E';
			loc = get_loc(target1, target2);
			char c = (char)waitKey(5);
			cout << l << "_" << ++tot << '_' << loc.x << '_' << loc.y << ' ';
			if (c == 27) break;
		}
		l++;
		cout << l << endl;
	}
	//main procedure
	return 0;
}
