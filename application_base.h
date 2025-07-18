#pragma once

#include "bejzak_engine/sources/command_buffer/command_buffer.h"
#include "bejzak_engine/sources/debug_messenger/debug_messenger.h"
#include "bejzak_engine/sources/instance/instance.h"
#include "bejzak_engine/sources/logical_device/logical_device.h"
#include "bejzak_engine/sources/physical_device/physical_device.h"
#include "bejzak_engine/sources/swapchain/swapchain.h"
#include "bejzak_engine/sources/window/window.h"
#include "bejzak_engine/sources/pipeline/shader_program.h"

class ApplicationBase {
protected:
    std::unique_ptr<Instance> _instance;
#ifdef VALIDATION_LAYERS_ENABLED
    std::unique_ptr<DebugMessenger> _debugMessenger;
#endif // VALIDATION_LAYERS_ENABLED
    std::unique_ptr<Window> _window;
    std::unique_ptr<Surface> _surface;
    std::unique_ptr<PhysicalDevice> _physicalDevice;
    std::unique_ptr<LogicalDevice> _logicalDevice;
    std::unique_ptr<Swapchain> _swapchain;
    std::unique_ptr<CommandPool> _singleTimeCommandPool;
    std::unique_ptr<ShaderProgramManager> _programManager;

public:
    ApplicationBase();
    virtual ~ApplicationBase() = default;
    virtual void run() = 0;
};
