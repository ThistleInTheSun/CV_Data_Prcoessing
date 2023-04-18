#include <iostream>
using namespace std;


class OCSort {
public:
    void display();
private:
};

void OCSort::display() {
    cout <<"First display"<<endl;
}

extern "C" {
    OCSort obj;
    void display() {
        obj.display(); 
      }

}