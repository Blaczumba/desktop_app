#include "application.h"

#include <memory>
#include <print>

#include "bejzak_engine/common/util/engine_exception.h"
#include "bejzak_engine/vulkan/wrapper/util/check.h"
#include "bejzak_engine/common/file/standard_file_loader.h"

int main() {
  try {
	Application app(std::make_unique<StandardFileLoader>());
	app.run();
  }
  catch (const VkException& vkException) {
	  std::println("Vulkan exception occured with message: {} and VkResult code: {}. \n {}", vkException.what(), static_cast<std::int32_t>(vkException.getResult()), vkException.stackTrace());
  }
  catch (const EngineException& engineException) {
	  std::println("Vulkan exception occured with message: {}. \n {}", engineException.what(), engineException.stackTrace());
  }

  return EXIT_SUCCESS;
}