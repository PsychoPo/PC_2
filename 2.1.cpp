/*#include <fstream>
#include <iostream>
#include <omp.h>
#include <math.h>
#include <vector>
using namespace std;
typedef double(*TestFunctTempl)(long&);


double Pi_Posled(long num_steps) {
    double time = omp_get_wtime();

    long i;
    double step, pi, x, sum = 0.0;
    step = 1.0 / (double)num_steps;
    for (i = 0; i < num_steps; i++)
    {
        x = (i + 0.5) * step;
        sum = sum + 4.0 / (1.0 + x * x);
    }
    pi = step * sum;


    double time_end = omp_get_wtime();

    return time_end - time;
}

double Pi_Static(long num_steps) {
    double time = omp_get_wtime();

    long i;

    double step, pi, x, sum = 0.0;
    step = 1.0 / (double)num_steps;
#pragma omp parallel for schedule(static,num_steps/30) private(x) reduction(+:sum)

    for (i = 0; i < num_steps; i++)
    {
        x = (i + 0.5) * step;
        sum = sum + 4.0 / (1.0 + x * x);
    }
    pi = step * sum;



    double time_end = omp_get_wtime();

    return time_end - time;
}

double Pi_dynamic(long num_steps) {
    double time = omp_get_wtime();

    long i;
    double step, pi, x, sum = 0.0;
    step = 1.0 / (double)num_steps;

#pragma omp parallel for schedule(dynamic,num_steps/30) private(x) reduction(+:sum)

    for (i = 0; i < num_steps; i++)
    {
        x = (i + 0.5) * step;
        sum = sum + 4.0 / (1.0 + x * x);
    }
    pi = step * sum;



    double time_end = omp_get_wtime();

    return time_end - time;
}

double Pi_guided(long num_steps) {
    double time = omp_get_wtime();

    long i;
    double step, pi, x, sum = 0.0;
    step = 1.0 / (double)num_steps;
#pragma omp parallel for schedule(guided,num_steps/30) private(x) reduction(+:sum)


    for (i = 0; i < num_steps; i++)
    {
        x = (i + 0.5) * step;
        sum = sum + 4.0 / (1.0 + x * x);
    }
    pi = step * sum;



    double time_end = omp_get_wtime();


    return time_end - time;
}

double Pi_Section(long num_steps) {
    double time = omp_get_wtime();

    long i;
    double step, pi, x, sum1 = 0.0, sum2 = 0.0, sum3 = 0.0, sum4 = 0.0;

    int n;
    step = 1.0 / (double)num_steps;
#pragma omp parallel
    {
        //omp_set_num_threads(n = 4);
        n = omp_get_max_threads();

    }

    int nt = 0, nt1 = num_steps / n, nt2 = num_steps * 2 / n, nt3 = num_steps * 3 / n;
#pragma omp parallel sections private (x)
    {
#pragma omp section
        {
            for (i = nt; i < nt1; i++)
            {


                x = (i + 0.5) * step;
                sum1 += 4.0 / (1.0 + x * x);


            }
        }


#pragma omp section
        {
            for (i = nt1; i < nt2; i++)
            {


                x = (i + 0.5) * step;
                sum2 += 4.0 / (1.0 + x * x);


            }
        }


#pragma omp section
        {
            if (num_steps > 2)
            {
                for (i = nt2; i < nt3; i++)
                {


                    x = (i + 0.5) * step;
                    sum3 += 4.0 / (1.0 + x * x);


                }
            }

        }


#pragma omp section
        {
            if (num_steps > 3)
            {
                for (i = nt3; i < num_steps; i++)
                {


                    x = (i + 0.5) * step;
                    sum4 += 4.0 / (1.0 + x * x);


                }
            }
        }

    }

    pi = step * (sum1 + sum2 + sum3 + sum4);


    double time_end = omp_get_wtime();

    return time_end - time;
}

double TestPi_Posled(long& num_steps) {
    return Pi_Posled(num_steps);
}
double TestPi_Static(long& num_steps) {
    return Pi_Static(num_steps);
}
double TestPi_dynamic(long& num_steps) {
    return Pi_dynamic(num_steps);
}
double TestPi_guided(long& num_steps) {
    return Pi_guided(num_steps);
}
double TestPi_Section(long& num_steps) {
    return Pi_Section(num_steps);
}

double AvgTrustedInterval(double& avg, vector<double>& times, int& cnt)
{
    double sd = 0, newAVg = 0;
    int newCnt = 0;
    for (int i = 0; i < cnt; i++)
    {
        sd += (times[i] - avg) * (times[i] - avg);
    }
    sd /= (cnt - 1.0);
    sd = sqrt(sd);
    for (int i = 0; i < cnt; i++)
    {
        if (avg - sd <= times[i] && times[i] <= avg + sd)
        {
            newAVg += times[i];
            newCnt++;
        }
    }
    if (newCnt == 0) newCnt = 1;
    return newAVg / newCnt;
}
double TestIter(void* Funct, long num_steps)
{
    double curtime = 0, avgTime = 0, avgTimeT = 0, correctAVG = 0;
    int iterations = 100;
    vector<double> Times(iterations);

    for (int i = 0; i < iterations; i++)
    {
        curtime = (((TestFunctTempl)Funct)(num_steps)) * 1000;
        Times[i] = curtime;
        avgTime += curtime;
        cout << "+";
    }

    cout << endl;
    avgTime /= iterations;
    cout << "AvgTime:" << avgTime << endl;
    avgTimeT = AvgTrustedInterval(avgTime, Times, iterations);
    cout << "AvgTimeTrusted:" << avgTimeT << endl;
    return avgTimeT;
}
void test_functions(void** Functions, vector<string> fNames)
{
    int nd = 0;

    double times[4][5][3];
    for (int num_steps = 500000; num_steps <= 2000000; num_steps += 500000)
    {
        for (int threads = 1; threads <= 4; threads++)
        {
            omp_set_num_threads(threads);
            for (int alg = 0; alg <= 4; alg++)
            {
                if (threads == 1)
                {
                    if (alg == 0) {
                        times[nd][alg][0] = TestIter(Functions[alg], num_steps);
                        times[nd][alg][1] = times[nd][alg][0];
                        times[nd][alg][2] = times[nd][alg][0];
                    }
                }
                else
                {
                    if (alg != 0)
                    {
                        times[nd][alg][threads - 2] = TestIter(Functions[alg], num_steps);
                    }
                }
            }
        }
        nd++;
    }

    ofstream fout("output.txt");
    fout.imbue(locale("Russian"));
    for (int ND = 0; ND < 4; ND++)
    {
        switch (ND)
        {
        case 0:
            cout << "\n----------500000 количество итераций----------" << endl;
            break;
        case 1:
            cout << "\n----------1000000 количество итераций----------" << endl;
            break;
        case 2:
            cout << "\n----------1500000 количество итераций----------" << endl;
            break;
        case 3:
            cout << "\n----------2000000 количество итераций----------" << endl;
            break;
        default:
            break;
        }


        for (int alg = 0; alg < 5; alg++)
        {
            for (int threads = 1; threads <= 4; threads++)
            {
                if (threads == 1)
                {
                    if (alg == 0) {
                        cout << "Thread " << threads << " --------------" << endl;
                        cout << fNames[alg] << "\t" << times[ND][alg][0] << " ms." << endl;
                        fout << times[ND][alg][0] << endl;

                    }
                }
                else
                {
                    if (alg != 0)
                    {
                        cout << "Thread " << threads << " --------------" << endl;
                        cout << fNames[alg] << "\t" << times[ND][alg][threads - 2] << " ms." << endl;
                        fout << times[ND][alg][threads - 2] << endl;
                    }
                }
            }
        }
    }
    fout.close();
}

int main()
{
    void** Functions = new void* [5] { TestPi_Posled, TestPi_Static, TestPi_dynamic, TestPi_guided, TestPi_Section};
    vector<string> function_names = { "Consistent realization","Parallel realization FOR(static)",
        "Parallel realization FOR(dinamic)", "Parallel realization FOR(guided)", "Parallel realization Section" };

    test_functions(Functions, function_names);

    return 0;

}*/