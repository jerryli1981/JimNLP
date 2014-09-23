package edu.pengli.nlp.platform.algorithms.lda;

public class TopicAspectModel {
	//tets

	private int K; // number of topics
	private int V; // vocabulary size
	private int D; // number of documents
	private int A; // number of aspects d
	
	private double beta;
	private double alpha;
	private double omega;
	private double gamma0;
	private double gamma1;
	private double delta0;
	private double delta1;
	
	private double alpha_sum;
	private double beta_sum;
	private double omega_sum;
	private double gamma_sum;
	private double delta_sum;

	private int[][] z; // topic assignment for word of document z_{d,n}
	private int[][] l; // level assignment for word of document l_{d,n}
	private int[][] x; // route assignment for word of document x_{d,n}
	private int[][] y; // aspect assignment for word of document y_{d,n}
	
////////////////////////////////////////////////////////////////////////////////////
	//below variables for compute v_p
	// l=0 & x=0 : the word come from the background model
	private int[] v_B_cnt; // the number of times word v has been assigned to Background topic
	private int B_sum; // the total number of words assigned to Background topic

	// l=0 & x=1 : the word come from the aspect model
    private int[][] v_a_cnt; // the number of times word v has been assigned to aspect a
    private int[] a_sum; // the total number of words assigned to aspect a
    
	// l=1 & x=0 : the word come from the topic model
	private int[][] v_k_cnt; // the number of times word v has been assigned to topic k
    private int[] k_sum; // the total number of words assigned to topic k

    
    // l=1 & x=1 : the word come from the topic model and aspect model
    private int[][][] v_k_a_cnt; //the number of times word v belongs to aspect a has been assigned to topic k
    private int[][] k_a_sum; // the total number of words belongs to aspect a assigned to topic k
    
///////////////////////////////////////////////////////////////////////////////////////////////////////////    
	// below variables for compute k_p
	private int[][] d_k_cnt; // the number of words from document i assigned to topic k 
	private int[] d_sum; // the total number of words from document i
    
    //below variables for compute a_p
    private int[][] d_a_cnt; // the number of words from document i has been assigned to aspect a

    
    //below variables for compute l_p;
    private int[][] d_l_cnt; // the number of words from document i has been assigned to level l

    //below variables for compute x_p
    private int[][] l_x_B_cnt; // In level l and route x, the number of words assigned to background topic 
    private int[] l_B_sum; // In level l, the total number of words assigned to background topic
    private int[][][] l_x_k_cnt; // In level l and route x, the number of words has been assigned to topic k
    private int[][] l_k_sum; // In level l, the total number of words assigned to topic k
    
    
    private void sampling_z_y_l_x(int d, int v) {
    	
    	int topic = z[d][v];
		int aspect = y[d][v];
		int level = l[d][v];
		int route = x[d][v];
		
		// decrease the counts		
		if (level == 0 && route == 0) {
			v_B_cnt[v]--;
			B_sum--;
		}
		else if (level == 0 && route == 1) {
			v_a_cnt[v][aspect]--;
			a_sum[aspect]--;
		}		
		else if (level == 1 && route == 0) {
			v_k_cnt[v][topic]--;
			k_sum[topic]--;
			
		}
		else if (level == 1 && route == 1) {
			v_k_a_cnt[v][topic][aspect]--;
			k_a_sum[topic][aspect]--;
		}
		
		d_k_cnt[d][topic]--;
		d_sum[d]--;
		d_a_cnt[d][aspect]--;
		d_l_cnt[d][level]--;
		l_x_B_cnt[level][route]--;
		l_B_sum[level]--;
		l_x_k_cnt[level][route][topic]--;
		l_k_sum[level][topic]--;
		
    	// sample new value for level and route
		double pTotal = 0.0;
    	double[] p = new double[4];
    	double v_p = 0.0;
    	double l_p = 0.0;
    	double x_p = 0.0;

    	// l = 0, x = 0
    	p[0] = (d_l_cnt[d][0] + delta0) /(d_sum[d] + delta_sum) * 
    			(l_x_B_cnt[0][0] + gamma0)/(l_B_sum[0]+gamma0+gamma1) * 
    			(v_B_cnt[v] +omega)/(B_sum + omega_sum);
    	// l= 0,  x = 1
    	p[1] = (d_l_cnt[d][0] + delta0) /(d_sum[d] + delta_sum) * 
    			(l_x_B_cnt[0][1] + gamma0)/(l_B_sum[0]+gamma0+gamma1) *
    			(v_a_cnt[v][aspect] +omega)/(a_sum[aspect] + omega_sum);
    	// l = 1, x = 0
    	p[2] = (d_l_cnt[d][1] + delta0) /(d_sum[d] + delta_sum) * 
    			(l_x_k_cnt[1][0][topic]+ gamma0)/(l_k_sum[1][topic] + gamma_sum)*
    			(v_k_cnt[v][topic] +omega)/(k_sum[topic] + omega_sum);
    	//l = 1, x = 1
        p[3] = (d_l_cnt[d][1] + delta0) /(d_sum[d] + delta_sum) * 
        		(l_x_k_cnt[1][1][topic]+ gamma0)/(l_k_sum[1][topic] + gamma_sum)*
        		(v_k_a_cnt[v][topic][aspect] + omega)/(k_a_sum[topic][aspect] + omega_sum);
        
        pTotal = p[0]+p[1]+p[2]+p[3];
        
        double X = Math.random() * pTotal;
        double tmp = 0;
        for(int i=0; i< 4; i++){
        	tmp += p[i];
        	if(tmp > X){
        		if (i >= 2) level = 1;
				else level = 0;
        		if (i % 2 == 1) route = 1;
				else route = 0;
				break;
        	}
        }
        
    	// sample new value for topic
        pTotal = 0.0;
		p = new double[K];
		v_p = 0;
		double k_p = 0;

		if( level ==0){
			for(int k=0; k<K; k++){
				k_p = (d_k_cnt[d][k] + alpha)/(d_sum[d] + alpha_sum);
				p[k] = k_p;
			}
			
		}else if(level ==1 && route ==0){
            for(int k=0; k<K; k++){
            	k_p = (d_k_cnt[d][k] + alpha)/(d_sum[d] + alpha_sum);
            	v_p = (v_k_cnt[v][k] +omega)/(k_sum[k] + omega_sum);
            	p[k] = k_p * v_p;
				
			}
			
		}else if(level ==1 && route ==1){
            for(int k=0; k<K; k++){
            	k_p = (d_k_cnt[d][k] + alpha)/(d_sum[d] + alpha_sum);
				v_p = (v_k_a_cnt[v][k][aspect] + omega)/(k_a_sum[k][aspect] + omega_sum);
				p[k] = k_p * v_p;
			}
		}
        
		tmp = 0;
		X = Math.random() * pTotal;
		for(int i=0; i<K; i++){
			tmp += p[i];
			if(tmp>X){
				topic = i;
			
				break;
			}
		}
    	
    	// sample new value for aspect
		   pTotal = 0.0;
		   p = new double[A];
		   double a_p = 0;
		   v_p = 0;
		
			if(route ==0){
				for(int a=0; a<A; a++){
					a_p = (d_a_cnt[d][a] + beta)/(d_sum[d] + beta_sum);
					p[a] = a_p;
				}
				
			}else if(route ==1 && level ==0){
                for(int a=0; a<A; a++){
                	a_p = (d_a_cnt[d][a] + beta)/(d_sum[d] + beta_sum);
                	v_p = (v_a_cnt[v][a] +omega)/(a_sum[a] + omega_sum);
                	p[a] = a_p * v_p;
				}
				
			}else if(route ==1 && level ==1){
                for(int a=0; a<A; a++){
                	a_p = (d_a_cnt[d][a] + beta)/(d_sum[d] + beta_sum);
                	v_p = (v_k_a_cnt[v][topic][a] + omega)/(k_a_sum[topic][a] + omega_sum);
                	p[a] = a_p * v_p;
				}
			}
	    tmp = 0;
		X = Math.random() * pTotal;
		for(int i=0; i<A; i++){
			tmp +=p[i];
			if(tmp > X){
				aspect = i;
				break;
			}
		}
		
    	// increase the counts
		if (level == 0 && route == 0) {
			v_B_cnt[v]++;
			B_sum++;
		}
		else if (level == 0 && route == 1) {
			v_a_cnt[v][aspect]++;
			a_sum[aspect]++;
		}		
		else if (level == 1 && route == 0) {
			v_k_cnt[v][topic]++;
			k_sum[topic]++;
			
		}
		else if (level == 1 && route == 1) {
			v_k_a_cnt[v][topic][aspect]++;
			k_a_sum[topic][aspect]++;
		}
		
		d_k_cnt[d][topic]++;
		d_sum[d]++;
		d_a_cnt[d][aspect]++;
		d_l_cnt[d][level]++;
		l_x_B_cnt[level][route]++;
		l_B_sum[level]++;
		l_x_k_cnt[level][route][topic]++;
		l_k_sum[level][topic]++;
		
		// set new assignments
		z[d][v] = topic;
		l[d][v] = level;
		y[d][v] = aspect;
		x[d][v] = route;
    }
    
}
