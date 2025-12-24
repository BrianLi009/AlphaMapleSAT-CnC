#include <unordered_map>

class ProgressTracker {
    private:
        struct Node {
            int rank;
            double progress;
            Node *next;
            Node *prev;
        };
        Node* front = 0;
        Node* back = 0;
        std::unordered_map<int, Node*> map;
    public:
        int size() { return map.size(); };
        int pop();
        void update(int rank, double progress);
};

int ProgressTracker::pop() {
    if (!front) { return 0; }
    Node* node = front;
    if (!front->next) { 
        front = 0;
        back = 0;
    } else {
        front->next->prev = 0;
        front = front->next;
    }
    int rank = node->rank;
    delete node;
    map.erase(rank);
    return rank;
}

void ProgressTracker::update(int rank, double progress) {
    if (map.find(rank) != map.end()) {
        Node* node = map[rank];
        if (front == node) { 
            pop();
        } else if (back == node) {
            back->prev->next = 0;
            back = back->prev;
            map.erase(rank);
            delete node;
        } else {
            node->next->prev = node->prev;
            node->prev->next = node->next;
            map.erase(rank);
            delete node;
        }
    }
    if (progress) {
        Node* node = new Node();
        node->rank = rank;
        node->progress = progress;
        node->next = 0;
        node->prev = back;
        if (front) {
            back->next = node;
        } else {
            front = node;
        }
        back = node;
        map[rank] = node;
    }
}