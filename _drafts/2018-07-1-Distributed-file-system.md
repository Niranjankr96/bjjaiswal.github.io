

## Distributed File Systems and why?
There are two option of storing big data
1. Have Big capacity node and
   *  also known as scale up, or vertical scaling.
   * More data == more storage hard-disk
   <figure>
     <div style="text-align:center">
       <img src="/assets/img/Big-data-notes/week1/scale-in.png" alt="scale-in"/>
     </div>
   </figure>
2. Store data in collective of nodes
   * Scale Out or horizontal Scale In
   * more data == more commodity Knowledge
   * <figure>
     <div style="text-align:center">
       <img src="/assets/img/Big-data-notes/week1/scale-out.png" alt="scale-out"/>
       <figcaption> Artificial brain. </figcaption>
     </div>

* One node will get out of service in three years on average. Thus with a cluster of 1,000 nodes you will get one pillar each day, approximately.

*  Each scaling approach has its own pros and cons. Accessing data you usually get lower latency with vertical scaling. You get higher latency with horizontal scale but you can build a bigger storage of a commodity service, and that is exactly the approach you will be dealing with during our courses.

> Google File System (GFS) : scalable distributed file system with a good level of full tolerance running on inexpensive commodity hardware.     

GFS key components:
* component failures are a norm rather than an exception.
  * all of the store data is duplicated, or more technically, replicated
*  equal distribution of space usage on different machines aka even spaced utilization
<figure>
  <div style="text-align:center">
    <img src="/assets/img/Big-data-notes/week1/replication.png" alt="scale-out"/>
    <figcaption> Replication over distributed system </figcaption>
  </div>

* write once read many data patterns
  * not allowed to modify files in the middle, as it dramatically simplifies API and internal implementation of a distributed file system.

`Metadata:` In the distributed file system, along with a local file systems, you should also have metadata.  Metadata includes administrative information about creation time, access properties, and so on. To request this metadata with minimal latency, you need master node, which stores all metadata in memory.
*   <figure>
  <div style="text-align:center">
    <img src="/assets/img/Big-data-notes/week1/GFS-masternode.png" alt="scale-out"/>
    <figcaption> Replication over distributed system </figcaption>
  </div>
  </figure>

> Hadoop Distributed File System, or HDFS for short, is an open source implementation of Google File System.

#### HDFS
* HDFS client provides command line interface to communicate with that distributed file system.
  * No need to write any code in program languages to access data
*  Available of RPC and option to access data via HTTP protocol.

##### How to read file from HDFS
<figure>
 <div style="text-align:center">
   <img src="/assets/img/Big-data-notes/week1/read-file-in-hdfs.png" alt="scale-out"/>
   <figcaption> Replication over distributed system </figcaption>
 </div>
 </figure>
* First, you request name node to get information about file blocks' locations.
  * These blocks are distributed over different machines, but all of this complexity is hidden behind HDFS API.
* User only sees a continuous stream of data
  * if at some point a datanode you retrieve data from died you get this data from another replica without bothering users about it
* Get data from the closest machine.
   * Closeness depends on the physical distance and unpredictable system load such as metric overutilization
   *  <figure>
    <div style="text-align:center">
      <img src="/assets/img/Big-data-notes/week1/distance.png" alt="scale-out"/>
      <figcaption> Replication over distributed system </figcaption>
    </div>
    </figure>
   * if data is available on the same machine, distance == 0
   * If a datanode is located in the same rack, then the distance is two.
   * If you are going to read data from another rack, then the distance is equal to four.
   * If the data is allocated in another data center, then there will be delivery overhead. and distance == 6

#### Writing a File in HDFS
`Redundancy model`
* When you write a block of data into HDFS, Hadoop distributes replicas over the storage. * The first replica is usually located on the same node if you write data from a DataNode machine.
  * <figure>
    <div style="text-align:center">
      <img src="/assets/img/Big-data-notes/week1/write-files-in-hdfs.png" alt="scale-out"/>
      <figcaption> Replication over distributed system </figcaption>
    </div>
    </figure>
* Otherwise, the first DataNode to put replica is chosen by random.    
* The second replica is usually placed in a different rack.
  * You find if this racks goes down, for instance because of power supply problems, then you will still be able to access data from another rack.
* The third replica is located on a different machine in the same rack as the second replica.
  * You don't pay for extra between rack network utilization because the third replica is copied from the second data node,

#### Data Flow in HDFS
* First, HDFS client request and name node via RPC protocol.
* The name node validates if you have rights to create a file and there are no naming conflicts.
* After that, HDFS client requests a list of datanodes to put a fraction of blocks of the file.
* These datanodes form a pipeline as your first client sends packets of data to the closest datanode.
* The later one transfers copies of packets through a datanode pipeline. As soon as packet is [INAUDIBLE] on all of the datanodes, datanodes send acknowledgment packets back.

* `What If something goes wrong`, then if the client closes the datanode pipeline, marks the misbehaving datanode bad and requests a replacement for the bad datanodes from a name node. So a new data node pipeline will be organized, and the process of writing the file to HDFS will continue. Again, all this happens transparently to a user. HDFS hides all this complexity for you.

* <figure>
  <div style="text-align:center">
    <img src="/assets/img/Big-data-notes/week1/failure-flow.png" alt="scale-out"/>
    <figcaption> Replication over distributed system </figcaption>
  </div>
  </figure>
*  Datanode serves a state machine for each block. Whenever a datanode recovers from its own failure, or failures of other datanodes in a pipeline, you can be sure that all the necessary replicas will be recovered. And unnecessary ones will be removed.

### Block and Replicas Recovery mode
* Replica is a physical data storage on a data node. There are usually several replicas with the same content on different data nodes.

* Block is a meta-information storage on a name node and provides information about replica's locations and their states. Both replica and block have their own states.

* If replica is in a finalized state then it means that the content of this replica is frozen. IF not finalized then meta-information for this block on name node is aligned with all the corresponding replica's states and data.
  * For instance you can safely read data from any data node and you will get exactly the same content. This property preserves read consistency.
*   
