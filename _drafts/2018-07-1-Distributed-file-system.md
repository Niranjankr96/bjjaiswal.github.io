

## Distributed File Systems and why?
There are two option of storing big data

| BIG Capacity Node      | Collection of Nodes           |
| ------------- |:-------------:|
| ![](/assets/img/Big-data-notes/week1/scale-in.png)|![](/assets/img/Big-data-notes/week1/scale-out.png)               |
|  also known as scale up, or vertical scaling.   | Scale Out or horizontal Scale In |
| More data == more storage hard-disk     |   more data == more commodity Knowledge   |  

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

### How to read file from HDFS
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

<figure>
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

<figure>
  <div style="text-align:center">
    <img src="/assets/img/Big-data-notes/week1/failure-flow.png" alt="scale-out"/>
  </div>
  </figure>

* Datanode serves a state machine for each block. Whenever a datanode recovers from its own failure, or failures of other datanodes in a pipeline, you can be sure that all the necessary replicas will be recovered. And unnecessary ones will be removed.

#### What have we learnt till now?
  * what vertical and horizontal scaling is?
  * server roles in HDFS
  * how topology affects replica placement?
  * what chunk / block size is used for?
  * how HDFS client reads and writes data?


### HDFS- Block and Replica States, Recovery Process.

* Replica is a physical data storage on a data node. There are usually several replicas with the same content on different data nodes.
* Block is a meta-information storage on a name node and provides information about replica's locations and their states. Both replica and block have their own states.
<figure>
  <div style="text-align:center">
    <img src="/assets/img/Big-data-notes/week1/finalized.png" alt="scale-out"/>
  </div>
  </figure>

* If replica is in a finalized state then it means that the content of this replica is frozen. IF not finalized then meta-information for this block on name node is aligned with all the corresponding replica's states and data.
  * For instance you can safely read data from any data node and you will get exactly the same content. This property preserves read consistency.

  <figure>
    <div style="text-align:center">
      <img src="/assets/img/Big-data-notes/week1/GS.png" alt="scale-out"/>
    </div>
    </figure>

* Each block of data has a version number called Generation Stamp(GS).
  * For finalized replicas, all blocks have same GN and icreases over time. It happens during error recovery process or during data appending to a block.

  <figure>
    <div style="text-align:center">
      <img src="/assets/img/Big-data-notes/week1/RBW.png" alt="scale-out"/>
    </div>
    </figure>

* `State RBW ----> Replica Being Written to.`
  * RBW is the state of the last block of an open file or a file which was reopened for appending.
  * During this state different data nodes can return to use a different set of bytes. In short, bytes that are acknowledged by the downstream data nodes in a pipeline are visible for a reader of this replica.
  * Data node on disk data and name node meta-information may not match.
  * `In case of any failure` data node will try to preserve as many bytes as possible.
  * It is a design goal called `data durability`.

  <figure>
    <div style="text-align:center">
      <img src="/assets/img/Big-data-notes/week1/RWR.png" alt="scale-out"/>
    </div>
    </figure>

* `State RWR ----> Replica Waiting to be Recovered.`
  * It is a state of all Being Written replicas after data node failure and recovery. For instance, after a system reboot or after Pacer.sys or BSOD, which are quite likely from a programming point of view.
  * RWR replicas will not be in any data node pipeline and therefore will not receive any new data packets. So they either become outdated and should be discarded, or they will participate in a special recovery process called a `lease recovery` if the client also dies.
  * HDFS client requests a lease from a name node to have an exclusive access to write or append data to a file.
  * In case of HDFS client lease expiration, replica transition to a RUR state.

> Lease expiration usually happens during the client's site failure.

<figure>
  <div style="text-align:center">
    <img src="/assets/img/Big-data-notes/week1/RUR.png" alt="scale-out"/>
  </div>
  </figure>


* `State RUR ---> Replica Under under recovery`  
  * A Hadoop administrator can spawn a process of data re-balancing or a data engineer can request increasing of the replication factor of data for the sake of durability. As data grows and different nodes are added or removed from a cluster, data can become unevenly distributed over the cluster nodes.  
  <figure>
    <div style="text-align:center">
      <img src="/assets/img/Big-data-notes/week1/Temporary.png" alt="scale-out"/>
    </div>
    </figure>
  * In  these cases new generated replicas will be in a state called `temporary`.
  * It is same state as RBW, however, this data is not visible to user unless finalized.
  * `In case of failure`, the whole chunk of data is removed without any intermediate recovery state.
  * In addition to the replica transition table, a name node block has its own collection of states and transitions.

* `Block State Transition`
   <figure>
  <div style="text-align:center">
    <img src="/assets/img/Big-data-notes/week1/Block-state.png" alt="scale-out"/>
  </div>
  </figure>

  * Different from data node replica states, a block state is stored in memory, it doesn't persist on any disk.

* `Under Construction State`
  * a user opens a file for writing or for appending name nodes, name node creates the corresponding block with the `under_construction_state`.
  * It is always the last block of a file, it's length and generation stamp are mutable.
  <figure>
    <div style="text-align:center">
      <img src="/assets/img/Big-data-notes/week1/under_construction.png" alt="scale-out"/>
    </div>
    </figure>
  * Name node block keeps track of right pipeline. It means that it contains information about all RBW and RWR replicas. It is quite vindictive and watches every step.
  <figure>
    <div style="text-align:center">
      <img src="/assets/img/Big-data-notes/week1/under_recovery.png" alt="scale-out"/>
    </div>
    </figure>
  * Replicas transitions from RWR to recovery RUR state when the client dies. Even more generally it happens when a client's lease expires. Consequently, the corresponding block transitions from `under_construction to under_recovery` state.
  * The `under construction block` transitions to a committed state when a client successfully requests name node to close a file or to create a new consecutive block of data.
* `Committed State`  
<figure>
  <div style="text-align:center">
    <img src="/assets/img/Big-data-notes/week1/recovery.png" alt="scale-out"/>
  </div>
  </figure>
  * The `committed state` means that there are already some finalized replicas but not all of them.
  * For this reason in order to serve a read request, the committed block needs to keep track of RBW replicas, until all the replicas are transitioned to the finalized state and HDFS client will be able to close the file. It has to retry it's requests.
*  `Final complete state` of a block
<figure>
  <div style="text-align:center">
    <img src="/assets/img/Big-data-notes/week1/complete.png" alt="scale-out"/>
  </div>
  </figure>
  * It is a state where all the replicas are in the finalized state and therefore `they have identical visible length and generation stamps`. Only when all the blocks of a file are complete the file can be closed.
  * ` In case of name node restart`, it has to restore the open file state. All the blocks of the un-closed file are loaded as complete except the last block which is loaded as under construction. Then recovery procedures will start to work.
  * There are several types of Recovery procedures:
     * replica recovery,
     * block recovery,
     * lease recovery, and
     * pipeline recovery.

#### BLOCK recovery
<figure>
  <div style="text-align:center">
    <img src="/assets/img/Big-data-notes/week1/block-recovery.png" alt="scale-out"/>
  </div>
  </figure>
* NameNode has to ensure that all of the corresponding replicas of a block will transition to a common state logically and physically i.e. all the correspondent replicas should have the same on disk content.
* To accomplish it, NameNode chooses a primary datanode called PD in a design document.
* PD should contain a replica for the target block.
* PD request from a NameNode, a new generation stamp, information and location of other replicas for recovery process.
* PD context each relevant DataNodes to participate in the replica recovery process.
>Replica recover process includes aborting active clients right into a replica. Aborting the previous replica of block recovery process, and participating in final replica size agreement process.

* During this phase, all the necessary information or data is propagated through the pipeline.
* At last, PD notifies NameNode about the result, success or failure. In case of failure, NameNode could retry block recovery process.

[![Everything Is AWESOME](//img.youtube.com/vi/StTqXEQ2l-Y/0.jpg)](//www.youtube.com/watch?v=StTqXEQ2l-Y "Everything Is AWESOME")
