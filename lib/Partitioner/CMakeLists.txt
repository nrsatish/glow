add_library(Partitioner
	      PartitionerUtils.cpp
	      Partitioner.cpp)

target_link_libraries(Partitioner
                      PRIVATE
                        Graph)
