#include <iostream>
#include <fstream>
#include <string>
#include <map>
#include <vector>
#include <algorithm>
#include <cmath>
#include <mpi.h>
#include <pthread.h>
#include <string.h>
#include <numeric>

using namespace std;

typedef pair<string, int> Item;
typedef pair<int, string> Item2;
typedef pair<string,vector<int>> Item3;
ofstream logg;

typedef struct map_arg{
    int chunkid;
    string s;
    int reducer;
}MAPARG;

typedef struct map_arg2{
    int redu_taskid;
    string name;
    string dir;
}MAPARG2;

void* mapper_task(void* arg);
void split(vector<Item2> &split_result, string line, int chunkid);
void map_func(vector<Item> &keyValuePair, vector<Item2> &split_result);
int partition(Item keyValuePair, int num);
void sort_f(vector<Item> &reduce_value,bool ascend);
void group(vector<Item3> &group_result, vector<Item> reduce_value);
void* reducer_task(void* arg);
void red(vector<Item3> group_result,vector<Item> &red_result);
void output(vector<Item> red_result,int redu_taskid,string name, string output_dir);

int main(int argc, char **argv){
    string job_name = string(argv[1]);
    int num_reducer = stoi(argv[2]);
    int delay = stoi(argv[3]);
    string input_filename = string(argv[4]);
    int chunk_size = stoi(argv[5]);
    string locality_config_filename = string(argv[6]);
    string output_dir = string(argv[7]);

    logg.open(job_name+"-"+"log.out");
    int rank, node;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &node);

    cpu_set_t cpu_set;
    sched_getaffinity(0, sizeof(cpu_set), &cpu_set);
    int ncpus = CPU_COUNT(&cpu_set);

    // read words file
    ifstream input_file(input_filename);
    string line;
    vector<string> chunk;
    int cnt = 1;
    while (getline(input_file, line))
    {
        if(cnt == 1){
            chunk.push_back(line);
        }
        else if(cnt <= chunk_size){
            chunk[chunk.size()-1] += line;
            if(cnt == chunk_size){
                cnt = 1;
                continue;
            }
        }
        cnt++;
    }
    input_file.close();

    //JOBTRACKER
    if(rank == 0){
        logg <<time(nullptr)<<",Start_Job,"<<job_name<<","<<node<<","<<ncpus<<","<<num_reducer<<","<<delay<<","<<input_filename<<","<<chunk_size<<","<<locality_config_filename<<","<<output_dir<<endl;
        double total_time = MPI_Wtime();

        //read locality table
        map<int, int> locality_table;
        input_file.open(locality_config_filename);
        while(getline(input_file,line)){
            int chunk_id, node_id, pos = line.find(" ");
            chunk_id = stoi(line.substr(0, pos));
            node_id = stoi(line.substr(pos,line.length()));
            locality_table[chunk_id] = node_id % (node-1) + 1;
        }
        input_file.close();
        
        //clear file
        ofstream out;
        for(int i = 0;i < num_reducer;i++){
            out.open(to_string(i)+"_mapper_result.txt",ios::trunc);
            out.close();
        }

        //Generate mapper tasks and insert them into a scheduling queue, where the taskID is the same as the chunkID.
        bool task1[locality_table.size()] = {false}; //default false => unfinished
        int req_node;
        int first = 0 , lo = 0;
        while(!(first == -1 && lo == -1)){ //will terminate until all task distributed
            for(int i = 1; i < node ; i++){ //process id
                MPI_Recv(&req_node, 1, MPI_INT, i, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                first = -1;
                lo = -1;
                for(int j = 0; j < locality_table.size(); j++){
                    if(first == -1 && !task1[j]){
                        first = j+1;
                    }
                    if(locality_table[j+1] == req_node && !task1[j]){
                        lo = j+1;
                        break;
                    }
                }
                if(lo == -1 && first != -1){ // return first task
                    logg<<time(nullptr)<<",Dispatch_MapTask,"<<first<<","<<i<<endl;
                    MPI_Send(&first, 1, MPI_INT, i, 2, MPI_COMM_WORLD);
                    task1[first-1] = true;
                }
                else if(lo != -1){ // return first task with data locality
                    logg<<time(nullptr)<<",Dispatch_MapTask,"<<lo<<","<<i<<endl;
                    MPI_Send(&lo, 1, MPI_INT, i, 2, MPI_COMM_WORLD);
                    task1[lo-1] = true;
                }
                else{ //return -1 to tasktracker to tell all mapper tasks are distributed
                    break;
                }
            }
        }
        for(int i = 1;i<node;i++){
            MPI_Send(&lo, 1, MPI_INT, i, 2, MPI_COMM_WORLD);
        }
        cout<<"all mapper task distributed\n";

        int done;
        int id, sec, cnt = 0;
        bool node_map[node+1] = {false};
        while(cnt < node-1){
            for(int i = 1; i < node ; i++){ //process id
                if(!node_map[i]){
                    MPI_Recv(&id, 1, MPI_INT, i, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    cout<<"id = "<<id<<endl;
                    if(id != -1){
                        MPI_Recv(&sec, 1, MPI_INT, i, 7, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                        logg << time(nullptr) <<",Complete_MapTask,"<<id<<","<<sec<<endl;
                    }
                    else if(id == -1){
                        node_map[i]=true;
                        cnt++;
                    }
                }
            }
        }
        
        //jobtracker know all mapper done
        cout<<"mapper done\n";

        //distributing reducing task
        int re = 0;
        while(re < num_reducer){
            for(int i = 1; i < node ; i++){ //process id
                MPI_Recv(&req_node, 1, MPI_INT, i, 4, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                if(re < num_reducer){
                    logg<<time(nullptr)<<",Dispatch_ReduceTask,"<< re + 1 <<","<<i<<endl;
                    MPI_Send(&re, 1, MPI_INT, i, 5, MPI_COMM_WORLD);
                    re++;
                }
            }
        }
        int k = -1;
        for(int i = 1; i < node ; i++){
            MPI_Send(&k, 1, MPI_INT, i, 5, MPI_COMM_WORLD);
        }
        cout<<"all reducer task distributed\n";

        bool node_red[node+1] = {false};
        cnt = 0;
        while(cnt < node-1){
            for(int i = 1; i < node ; i++){ //process id
                if(!node_red[i]){
                    MPI_Recv(&id, 1, MPI_INT, i, 6, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    cout<<"id = "<<id<<endl;
                    if(id != -1){
                        MPI_Recv(&sec, 1, MPI_INT, i, 8, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                        logg << time(nullptr) <<",Complete_ReduceTask,"<< id+1 <<","<<sec<<endl;
                    }
                    else if(id == -1){
                        node_red[i]=true;
                        cnt++;
                    }
                }
            }
        }

        logg<<time(nullptr)<<",Finish_Job,"<< int((MPI_Wtime()-total_time)*1000)<<endl;
        /**********this is jobtracker end***********/
    }
    //TASKTRACKER
    else {
        int nodeid = rank, chunkid, map_cnt = 0;
        vector <int> chunkid_list;
        pthread_t thread[ncpus-1],reducer;
        MAPARG t;
        int k = 0;
        string s;
        void *ss;
        
        //get chunk id
        while(1){
            MPI_Send(&nodeid, 1, MPI_INT, 0, 1, MPI_COMM_WORLD);
            MPI_Recv(&chunkid, 1, MPI_INT, 0, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            cout<<"nodeid = "<<nodeid<<" recv:"<<chunkid<<endl;
            if(chunkid == -1) break; //get all mapper task
            else{
                chunkid_list.push_back(chunkid);
                map_cnt ++;
            }
        }
        int wait_s, wait_e;

        int finishid = 0;
        int tmp;
        double times[chunkid_list.size()];
        /**********this is mapper start***********/
        for(int i = 0 ;i < chunkid_list.size();i++){

            //delay
            // wait_s = MPI_Wtime();
            // wait_e = MPI_Wtime();
            // while(wait_e - wait_s < delay){ //wait for D seconds to simulate locality
            //     wait_e = MPI_Wtime();
            // }

            t.chunkid = chunkid_list[i]-1;
            t.s = chunk[t.chunkid];
            t.reducer = num_reducer;
            if(i >= ncpus - 1){
                pthread_join(thread[i%(ncpus-1)],&ss);
                tmp = int((MPI_Wtime()-times[finishid])*1000);
                MPI_Send(&chunkid_list[finishid], 1, MPI_INT, 0, 3, MPI_COMM_WORLD);
                MPI_Send(&tmp, 1, MPI_INT, 0, 7, MPI_COMM_WORLD);
                finishid++;
            }
            times[i] = MPI_Wtime();
            pthread_create(&thread[i%(ncpus-1)],NULL,mapper_task,&t);
            cout<<"nodeid = "<<nodeid<<" create a mapper thread\n";
        }
        
        for(;finishid<chunkid_list.size();finishid++){
            pthread_join(thread[finishid%(ncpus-1)],&ss);
            tmp = int((MPI_Wtime()-times[finishid])*1000);
            MPI_Send(&chunkid_list[finishid], 1, MPI_INT, 0, 3, MPI_COMM_WORLD);
            MPI_Send(&tmp, 1, MPI_INT, 0, 7, MPI_COMM_WORLD);
        }
        /**********this is mapper end***********/

        int done = -1;
        MPI_Send(&done, 1, MPI_INT, 0, 3, MPI_COMM_WORLD);
        //let jobtracker know all mapper done

        //get reducer task
        int reducerid;
        vector<int> reducer_list;
        while(1){
            MPI_Send(&nodeid, 1, MPI_INT, 0, 4, MPI_COMM_WORLD);
            MPI_Recv(&reducerid, 1, MPI_INT, 0, 5, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            cout<<"nodeid = "<<nodeid<<" recv redu:"<<reducerid<<endl;
            if(reducerid == -1)break;
            reducer_list.push_back(reducerid);
        }
        

        //create reducer thread
        MAPARG2 r;
        double tmp_time;
        for(int i = 0 ; i <reducer_list.size();i++){
            r.redu_taskid = reducer_list[i];
            r.name = job_name;
            r.dir = output_dir;
            pthread_create(&reducer,NULL,reducer_task,&r);
            cout<<"nodeid = "<<nodeid<<" create a reducer thread\n";
            tmp_time = MPI_Wtime();
            pthread_join(reducer,&ss);
            tmp = int((MPI_Wtime()-tmp_time)*1000);

            MPI_Send(&reducer_list[i], 1, MPI_INT, 0, 6, MPI_COMM_WORLD);
            MPI_Send(&tmp, 1, MPI_INT, 0, 8, MPI_COMM_WORLD);
            cout<<"nodeid = "<<nodeid<<" finish a reducer thread\n";
        }


        cout<<"finished reduce\n";
        MPI_Send(&done, 1, MPI_INT, 0, 6, MPI_COMM_WORLD);
        /**********this is tasktracker end***********/
    }

    MPI_Finalize();
    logg.close();
    return 0;
}

void* reducer_task(void* arg){

    //read intermediate key-value pair
    MAPARG2 *sp = (MAPARG2*)arg;
    int redu_taskid = sp->redu_taskid;
    string name = sp->name;
    string output_dir = sp->dir;

    ifstream p;
    string keyValue;
    int intermediate;
    Item tt;
    vector<Item> reduce_value;
    bool ascend = true;

    p.open(to_string(redu_taskid)+"_mapper_result.txt",ios::in);
    while(p>>keyValue>>intermediate){
        tt = make_pair(keyValue,intermediate);
        reduce_value.push_back(tt);
    }
    p.close();

    //sort
    sort_f(reduce_value, ascend);
    // cout<<"reduce_list.size = "<<reduce_value.size()<<endl;

    //group
    vector<Item3> group_result;
    group(group_result,reduce_value);

    //reduce
    vector<Item> red_result;
    red(group_result,red_result);

    //output
    output(red_result,redu_taskid,name,output_dir);
    
    pthread_exit(NULL);
}

void output(vector<Item> red_result,int redu_taskid,string name, string output_dir){
    ofstream output;
    output.open("./"+output_dir+"/"+name+"-"+to_string(redu_taskid+1)+".out");
    for(int j = 0 ; j < red_result.size();j++){
        output << red_result[j].first <<" "<<red_result[j].second<<endl;
    }
    output.close();
}

void red(vector<Item3> group_result,vector<Item> &red_result){
    int sum;
    Item t;
    for(int i = 0 ;i< group_result.size();i++){
        sum = accumulate(group_result[i].second.begin(),group_result[i].second.end(),0);
        t = make_pair(group_result[i].first,sum);
        red_result.push_back(t);
    }
}

void group(vector<Item3> &group_result, vector<Item> reduce_value){
    //groupbykey
    Item3 ttt;
    vector<int> cnt;
    cnt.push_back(1);
    // cout<<"func:"<<reduce_value.size()<<endl;
    for(int i = 0 ; i < reduce_value.size();i++){
        if(group_result.size() == 0 || group_result[group_result.size()-1].first != reduce_value[i].first){
            ttt = make_pair(reduce_value[i].first, cnt);
            group_result.push_back(ttt);
        }else if(group_result[group_result.size()-1].first == reduce_value[i].first){
            group_result[group_result.size()-1].second.push_back(1);
        }
    }

    //group by first character of string
    // pair<char,vector<string>> tpp;
    // vector<pair<char,vector<string>>> vec;
    // vector<string> str;
    // for(int i = 0 ; i < reduce_value.size();i++){
    //     if(vec.size() == 0 || vec[vec.size()-1].first != reduce_value[i].first[0]){
    //         str.clear();
    //         str.push_back(reduce_value[i].first);
    //         tpp = make_pair(reduce_value[i].first[0], str);
    //         vec.push_back(tpp);
    //     }else if(vec[vec.size()-1].first == reduce_value[i].first[0]){
    //         vec[vec.size()-1].second.push_back(reduce_value[i].first);
    //     }
    // }

    //write to file
    // ofstream group("group.txt");
    // for(int i = 0 ; i < vec.size();i++){
    //     group<<vec[i].first<<" : ";
    //     for(int j = 0 ; j < vec[i].second.size(); j++){
    //         group<<vec[i].second[j]<<" ";
    //     }
    //     group<<endl;
    // }
    // group.close();
}

void sort_f(vector<Item> &reduce_value,bool ascend = true){
    if(ascend)
        sort(reduce_value.begin(),reduce_value.end());
    else
        sort(reduce_value.begin(),reduce_value.end(),greater<Item>());
    
    // for(int i = 0 ; i < reduce_value.size();i++){
    //     cout<<reduce_value[i].first<<" "<<reduce_value[i].second<<endl;
    // }
}

void* mapper_task(void* arg){
    MAPARG *sp = (MAPARG*)arg;
    int chunkid = sp->chunkid;
    string line = sp->s;
    int reducer_num = sp->reducer;
    vector<Item> keyValuePair;
    vector<Item2> split_result;

    //split into words
    split(split_result,line,chunkid);
   
    //map function
    map_func(keyValuePair,split_result);
    
    //partition function
    map<Item, int> hash_table;
    int val;
    for(int j = 0 ;j < keyValuePair.size();j++){
        val = partition(keyValuePair[j],reducer_num);
        hash_table[keyValuePair[j]] = val;
    }

    //write to local FS
    ofstream output;
    for(int i = 0;i < reducer_num;i++){
        output.open(to_string(i)+"_mapper_result.txt",ios::app);
        for(int j = 0 ; j < keyValuePair.size();j++){
            if(hash_table[keyValuePair[j]] == i)
                output << keyValuePair[j].first <<" "<<keyValuePair[j].second<<endl;
        }
        output.close();
    }
    pthread_exit(NULL);
}

void split(vector<Item2> &split_result, string line, int chunkid){
    string s;
    Item2 t;
    while(line.find(" ") != string::npos ){
        s = line.substr(0, line.find(" "));
        t = make_pair(chunkid,s);
        split_result.push_back(t);
        line.erase(0,line.find(" ") + 1);
    }
}

void map_func(vector<Item> &keyValuePair, vector<Item2> &split_result){
    Item t;
    for(int i=0;i<split_result.size();i++){
        t = make_pair(split_result[i].second,1);
        keyValuePair.push_back(t);
    }
}

int partition(Item keyValuePair, int num){
    // hash<string> str_hash;
    // return str_hash(keyValuePair.first) % num;
    return int(keyValuePair.first[0]) % num;

    // if(keyValuePair.first[0]>='A' && keyValuePair.first[0]<='Z'){

    // }
    // else{

    // }
}