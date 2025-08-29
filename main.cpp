#include "application.h"

#include <memory>

#include "bejzak_engine/common/file/standard_file_loader.h"

int main() {
  Application app(std::make_unique<StandardFileLoader>());
  app.run();

  return EXIT_SUCCESS;
}