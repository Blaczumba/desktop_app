## Manual for building the project
### Prerequisites
* Vulkan SDK installed in the OS
    1) Windows - https://www.lunarg.com/vulkan-sdk/
    2) Raspberry PI 4/5 (Video with installation) - https://www.youtube.com/watch?v=TLzFPIoHhS8&t=78s&ab_channel=NovaspiritTech
* CMake 3.19 (at least) installed in the OS
* Development libraries are located in the `external` folder - these are downloaded with repository

### Downloading all the necessary repositories
* git clone https://github.com/Blaczumba/desktop_app.git
* cd desktop_app
* git submodule update --init --recursive

### Building for Windows
* Install Visual Studio Community/Professional
* Open repository folder with Visual Studio
* Open CMakeLists.txt and save the file (ctrl + s)
* Download gltf sponza model from https://sketchfab.com/3d-models/sponza-0cbee5e07f3a4fae95be8b3a036abc91
* Put the "sponza" folder to desktop_app/assets/models (create folder if not existing)
* Download cubemap_yokohama_rgba.ktx from https://github.com/SaschaWillems/Vulkan-Assets/tree/a27c0e584434d59b7c7a714e9180eefca6f0ec4b/textures
* Put image to the desktop_app/assets/textures (create folder if not existing)
* On the toolbar next to the green triangle select the target to run (VulkanProject.exe or VulkanTests.exe)

### Building for Raspberry PI
* Clone the repository
* Go to the repository with `cd` command
* `$ cmake -B build -S .`
* `$ cd build`
* `$ make`
* `$ cd bin`
* `$ ./VulkanProject`