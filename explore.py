import pandas as pd
import numpy as np

def ki_squared_for_explore(train,threshold=.75):
    ''' 
    input train, the rest has been tailored for use in this project
    play with chi_list if you're looking to add or remove feature to investigate
    threshold drops features that don't hold significance?
    '''
    import seaborn as sns
    import scipy.stats as stats
    import matplotlib.pyplot as plt

    chi_list = ['is_highlight','is_timeline_work','is_public_domain','portfolio','artist_gender','has_artist_url',
                'object_wikidata_url_1_or_0','gallery_number_0','gallery_number_774','gallery_number_999','department_Ancient Near Eastern Art',
                'department_Arms and Armor','department_Arts of Africa, Oceania, and the Americas','department_Asian Art',
                'department_Costume Institute','department_Drawings and Prints','department_Egyptian Art','department_European Paintings',
                'department_European Sculpture and Decorative Arts','department_Greek and Roman Art','department_Islamic Art',
                'department_Medieval Art','department_Modern and Contemporary Art','department_Musical Instruments','department_Photographs',
                'department_Robert Lehman Collection','department_The American Wing','department_The Cloisters','department_The Libraries',
                'object_name_Book','object_name_Drawing','object_name_Fragment','object_name_Kylix fragment','object_name_Negative',
                'object_name_Other','object_name_Painting','object_name_Photograph','object_name_Piece','object_name_Print','culture_American',
                'culture_British','culture_China','culture_French','culture_Greek, Attic','culture_Italian','culture_Japan','culture_Japanese',
                'culture_Other','culture_Roman','culture_Unknown',
                'credit_line_Gift of Dietrich von Bothmer, Distinguished Research Curator, Greek and Roman Art, 2011','credit_line_Other',
                'credit_line_The Cesnola Collection, Purchased by subscription, 1874–76',
                'credit_line_The Jefferson R. Burdick Collection, Gift of Jefferson R. Burdick','credit_line_Walker Evans Archive, 1994',
                'medium_Albumen photograph','medium_Albumen silver print from glass negative','medium_Bronze','medium_Commercial color lithograph',
                'medium_Engraving','medium_Etching','medium_Faience','medium_Film negative','medium_Gelatin silver print','medium_Glass',
                'medium_Lithograph','medium_None','medium_Other','medium_Silk','medium_Terracotta','cluster_strong_yes','cluster_strong_no']

    column_names=chi_list


    chisqmatrix=pd.DataFrame(train,columns=column_names,index=column_names)
    chisqmatrix

    outercnt=0
    innercnt=0
    for icol in column_names:

        for jcol in column_names:

           mycrosstab=pd.crosstab(train[icol],train[jcol])
           #print (mycrosstab)
           stat,p,dof,expected=stats.chi2_contingency(mycrosstab)
           chisqmatrix.iloc[outercnt,innercnt]=round(p,3)
           cntexpected=expected[expected<5].size
           perexpected=((expected.size-cntexpected)/expected.size)*100

           #print(stat)
           #print(icol)
           #print(jcol)
           if perexpected<20:
                chisqmatrix.iloc[outercnt,innercnt]=2
           #print (perexpected) 
           if icol==jcol:
               chisqmatrix.iloc[outercnt,innercnt]=0.00
           #print (expected) 
           innercnt=innercnt+1
        #print (outercnt) 
        outercnt=outercnt+1
        innercnt=0



    import seaborn as sns
    import matplotlib.patches as patches

    high_chi = chisqmatrix.sum()[chisqmatrix.sum()>chisqmatrix.sum().quantile(threshold)].index.tolist()

    df_to_plot = chisqmatrix[chisqmatrix.index.isin(high_chi)][high_chi].copy()
    df_to_plot = df_to_plot.replace(0,.01)
    df_to_plot = df_to_plot.astype(np.float64)

    matrix = np.triu(df_to_plot)

    fig,ax = plt.subplots(figsize=(12,6))
    df_to_plot = df_to_plot.replace(0.01,0).round(1)
    sns.heatmap(df_to_plot, annot=True,linewidths=0.1, 
                cmap='coolwarm',mask=matrix)

    ax.add_patch(patches.Rectangle((0, 1),1.0,int(len(df_to_plot.index)+1),edgecolor='red',fill=False,lw=4) ) ##tuple is top-x,y
    plt.xticks(rotation=80)
    plt.title("Relations of Porportions for our most imbalanced variables (0 == Most Imbalanced, 1 == Most Balanced)")
    plt.show()

def chi2_for_feature(df,feature):
    ''' 
    input the df dataset, string of the target, and a list of featues to run through
    does a chi2 test for indepenance (proportionality) and plots the results
    no return
    '''
    # creates a column of the payments for easy analysis, runs a crosstab to put into a chi2 independancy test.
    # produces observed and expected values
    # returns the chi2 and pval for the whole set
    import scipy.stats as stats
    import matplotlib.pyplot as plt

    df1 = df[[feature,"is_highlight",]].groupby(feature).agg(["sum","count"]).droplevel(axis=1, level=0)

    chi2, p, degf, expected = stats.chi2_contingency(df1.iloc[:,-2:])

    alpha = .05
    H0 = (f"{feature.title().replace('_',' ')} is not different in the porportions of Is Highlight")
    H1 = (f"{feature.title().replace('_',' ')} is different in the porportions of Is Highlight")
    #print('Observed')
    #print(df1.values)
    #print('---\nExpected')
    dfexpected = pd.DataFrame(expected,columns=df1.columns,index=df1.index)
    #print(dfexpected.values)
    print(f'---\nchi^2 = {chi2:.4f}, p = {p:.5f}, degf = {degf}')
    if p>alpha:
        print(f"due to p={p:.5f} > α={alpha} we fail to reject our null hypothesis\n({H0})")
    else:
        print(f"due to p = {p:.5f} < α = {alpha} we reject our null hypothesis\n( ", '\u0336'.join(H0) + '\u0336' , ")")

    #plot the results
    height = (len(df1.index)//4+1)*2.4
    plt.figure(figsize=(18,height))
    plt.suptitle(f"Porportions in {feature.title().replace('_',' ')} for Is Highlight", fontsize=16, y=0.99)

    for x,col in enumerate(df1.T.columns):
            ax = plt.subplot(len(df1.index)//4+1,4,x+1)
            pd.concat({'Expected': dfexpected.T[col], 'Observed': df1.T[col]}, axis=1).iloc[0,].\
                plot.bar(color=["grey","pink"], edgecolor="black",ax=ax)
            ax.set_ylabel("Count")
            #ax.set_ylim([0,dfexpected["sum"].max()])
            #ax.set_yscale("log")
            ax.set_xticklabels(["Expected","Observed"],rotation=0)
            ax.set_title(f'{col} values') # Title with column name.
    plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=0.9,
                    top=(1-(.6/height)),
                    wspace=0.2,
                    hspace=0.4)
    plt.show()