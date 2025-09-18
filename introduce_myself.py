import streamlit as st

col1, col2 = st.columns(2)

with col1:
    with st.expander('자기소개'):   
     with st.popover('저를 소개합니다.'):
        st.write('한양대학교 데이터사이언스 전공 3학년으로 편입학한 김진욱이라고 합니다.\n')

        st.write('한학기동안 동아리 활동을 통해 많은 것을 얻어가고 싶습니다.')    

with col2:
    with st.expander('취미'):
     st.write('저는 게임과 영화와 책을 좋아합니다.')
     with st.popover("더보기"):
        tab1, tab2, tab3, tab4 = st.tabs(['게임','영화','책','음악'])
    with tab1:
        with st.popover('EA SPORTS FC'):
           st.write('FC(구 FIFA 시리즈)는 어릴때부터 꾸준히 즐겨오고 있는 게임입니다.')
        st.image('https://media.contentapi.ea.com/content/dam/ea/fifa/fifa-23/common/news/f23-featured-image-ea-play-trial.jpg.adapt.crop16x9.575p.jpg',width=400)

        with st.popover('LoL'):
           st.write('가끔 칼바람하면 재밌습니다.')
        st.image('https://scontent-ssn1-1.xx.fbcdn.net/v/t1.6435-9/59973952_2362475807130125_2563295675492073472_n.png?_nc_cat=108&ccb=1-7&_nc_sid=127cfc&_nc_ohc=EAtqAjWPfSwQ7kNvwF2yea2&_nc_oc=AdnTHvqJR446Eft91vA2cfRKvlHkJ8Do7VcFh5h6julnQZVFbk3QzP0biOYTcoqIx2A&_nc_zt=23&_nc_ht=scontent-ssn1-1.xx&_nc_gid=iVKsv9Wf1rQfj2q3pF8xIg&oh=00_AfaV98lMdXCcv6cxKJqWafS8IJfr5qXtVO_qrX9gOtzYhw&oe=68F2CBC7',width=400)

        with st.popover('Elden Ring'):
           st.write('유일하게 Steam 모든 도전과제를 깬 게임입니다. 플레이할때 정말 즐거웠던 게임.')
        st.image('https://image.api.playstation.com/vulcan/ap/rnd/202402/0817/114b1df9577098209a8bb8e45f4a009e201e9a2fa5113a06.png',width=400)

        with st.popover('Dark Souls3'):
           st.write('Elden Ring을 클리어하고 난 다음 아쉬움을 달랬던 게임입니다. 조금 오래돼서 그래픽이 뭔가 좀.')
        st.image('https://cdn.mos.cms.futurecdn.net/J5pxCRKSNAhwbdcbGitmdT-1200-80.jpg',width=400)            
    with tab2:
        with st.expander('Empire Strike Back'):
           st.write('스타워즈를 상징하는 작품')
        st.image('https://m.media-amazon.com/images/M/MV5BMTkxNGFlNDktZmJkNC00MDdhLTg0MTEtZjZiYWI3MGE5NWIwXkEyXkFqcGc@._V1_.jpg',width=400)

        with st.expander('Revenge of the Sith'):
           st.write('개봉한지 20년이 지났지만 비주얼은 아직까지도 최고인듯.')
        st.image('https://images.fandango.com/ImageRenderer/500/0/redesign/static/img/default_poster--dark-mode.png/0/images/masterrepository/Fandango/239973/starwarsepisodeiii-rerelease-posterart.jpg',width=400)

        with st.expander('Where are sen and chihiro?'):
           st.write('영화를 심도있게 감상해서 뭐라 형용해야될지는 모르겠지만 특유의 분위기가 있다.')
        st.image('https://upload.wikimedia.org/wikipedia/ko/b/bc/%EC%84%BC%EA%B3%BC_%EC%B9%98%ED%9E%88%EB%A1%9C%EC%9D%98_%ED%96%89%EB%B0%A9%EB%B6%88%EB%AA%85_%ED%8F%AC%EC%8A%A4%ED%84%B0.jpg',width=400)

        with st.expander('Inception'):
           st.write('피셔를 가스라이팅하는 과정을 다이나믹하게 담아냈다. 브금들이 아주 좋다.')
        st.image('https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQxRA-A7B7Cc08fHigWBJoTiGvPQgrFP1K5OA&s',width=400)

        with st.expander('반지의 선택'):
           st.write('반면교사도 어쨌든 교사다.(사실 영화는 안보고 나무위키 문서만 정독함)')
        st.image('https://i.namu.wiki/i/ejjZYbk393tI-PefOyCJz0EnhfHFC-2bXeBZpAdvAAUuC-EVySu_HGcPxryBOxdh-KuOMT7kXZPgaO2MVhp_Zw.webp',width=400)

        with st.expander('Clementine'):
           st.write('사실 얘도 나무위키 문서만 본듯.') 
        st.image('https://image.tving.com/ntgs/contents/CTC/caim/CAIM2100/ko/20250317/0506/M000345335.jpg/dims/resize/480',width=400) 

with tab3:
    with st.expander('은하수를 여행하는 히치하이커를 위한 안내서'):
       st.write('합본권이 1200page 가량 되는데 처음부터 끝까지 서술방식이 일관된다. 이 책을 읽고나서 오디세이아를 읽으니까 줄거리를 따라가기가 너무 쉽다.')
    st.image('https://contents.kyobobook.co.kr/sih/fit-in/400x0/pdt/9788970135472.jpg',width=400)

    with st.expander('아메리칸 프로메테우스'):
       st.write('밤새고 오펜하이머를 보러가서 영화관에서 자다만 와서 아쉬워서 산 책. 이 책 읽고 오펜하이머 2회차 가니까 더 재밌게 느껴졌다.')
    st.image('https://d2phebdq64jyfk.cloudfront.net/media/community/image/1e1e9f85-22fb-4939-baf3-25a0395b6957.JPEG',width=400)

    with st.expander('일론 머스크'):
       st.write('포브스 선정 전세계 부자 1위의 삶을 담은 책. 자극적인 행보때문에 빠와 까가 장난아니게 많지만, 이미 많은 부를 축적하였음에도  열심히 사는 그의 모습을 보면서 영감을 받아서 편입 공부를 시작하게 되었다.')
    st.image('https://contents.kyobobook.co.kr/pmtn/2023/book/230821_musk/bnM_e01_02.png',width=400)

    with st.expander('전설로 떠나는 월가의 영웅들'):
       st.write('SNP 500을 순수체급으로 이긴데에는 이유가 있는 것 같다. 난 도저히 이 사람처럼 할 자신이 없어서 SPDR 사기로 마음 먹었다.')
    st.image('https://mblogthumb-phinf.pstatic.net/MjAyMzExMTlfMjM1/MDAxNzAwMzQ2OTg4NzQ4.jqCAJCGz8CIBYmApi4I-NHHHVe8_DfZRfC6Li02YTIAg.CoPblXmncvav9Jsrmd9BYY5CKW2VFphQ4sRzbBqQVI0g.PNG.jjib2002/image.png?type=w800',width=400)

    with st.expander('파인만씨 농담도 잘하시네'):
       st.write('교양이 없어서 그런가 총균쇠 이런 책은 읽어도 이해가 잘 안되는데 이 책은 되게 재밌게 읽었다.')
    st.image('https://contents.kyobobook.co.kr/sih/fit-in/400x0/pdt/9788983710444.jpg',width=400)

with tab4:
    st.write('A new hope and End Credits')
    st.video('https://youtu.be/ZK52tEenER8?si=CcYt2ZxtO0iKe2Kb',start_time=42)

    st.write('Time')
    st.video('https://youtu.be/c56t7upa8Bk?si=_0uxoWgjaWKLtzj7',start_time=42)

    st.write('Promised Consort')
    st.video('https://youtu.be/OK5ObZ_vXsU?si=0e9V8CpnngHFbp1J',start_time=42)

    st.write('분홍신')
    st.video('https://youtu.be/XaZptY3DCxk?si=TkYUwaGcwNvIA8f1',start_time=42)

st.write('streamlitrun은 처음이라 조금 어렵네요;')