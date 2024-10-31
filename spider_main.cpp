#include <iostream>
#include <cstring>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>

#define PORT 8080

int sock = 0;
struct sockaddr_in serv_addr;

void initializeSocket() {
    if ((sock = socket(AF_INET, SOCK_STREAM, 0)) < 0) {
        std::cout << "Socket creation error" << std::endl;
        return;
    }

    serv_addr.sin_family = AF_INET;
    serv_addr.sin_port = htons(PORT);

    if (inet_pton(AF_INET, "127.0.0.1", &serv_addr.sin_addr) <= 0) {
        std::cout << "Invalid address / Address not supported" << std::endl;
        return;
    }

    if (connect(sock, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) < 0) {
        std::cout << "Connection Failed" << std::endl;
        return;
    }
}

void sendCommand(const char* command) {
    send(sock, command, strlen(command), 0);
}

int main() {
    initializeSocket();

    while (true) {
        char input;
        std::cout << "Enter command (F for forward, R for rotate right, Q to quit): ";
        std::cin >> input;

        if (input == 'Q' || input == 'q') {
            break;
        }

        sendCommand(&input);
    }

    close(sock);
    return 0;
}
